"""GNN-Mamba3 Dynamics: drop-in replacement for DLPDynamics.

Maintains the same forward()/sample() interface and output dict format.
Replaces ParticleSpatioTemporalTransformer with GNNMamba3Transformer.
Reuses ParticleFeatureProjection and ParticleFeatureDecoderDyn from LPWM.
"""

import sys
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from .gnn_mamba3_block import GNNMamba3Transformer
from .config import GNNMamba3Config


def _get_lpwm_imports():
    """Lazy import of LPWM modules to avoid heavy dependency chain at import time."""
    lpwm_path = os.path.join(os.path.dirname(__file__), '..', '..', 'baseline', 'lpwm', 'upstream')
    lpwm_path = os.path.abspath(lpwm_path)
    if lpwm_path not in sys.path:
        sys.path.insert(0, lpwm_path)
    from modules.modules import ParticleFeatureProjection, ParticleFeatureDecoderDyn, RMSNorm
    return ParticleFeatureProjection, ParticleFeatureDecoderDyn, RMSNorm


def reparameterize(mu, logvar):
    """Reparameterization trick: sample z = mu + eps * std."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class GNNMamba3Dynamics(nn.Module):
    """GNN-Mamba3 particle dynamics with the same interface as DLPDynamics.

    Keeps:
      - ParticleFeatureProjection (input encoding)
      - ParticleFeatureDecoderDyn (output decoding)
      - Context conditioning logic (adaln, actions)
    Replaces:
      - ParticleSpatioTemporalTransformer → GNNMamba3Transformer
    """

    def __init__(self,
                 features_dim,
                 bg_features_dim,
                 hidden_dim,
                 projection_dim,
                 n_head=8,
                 n_layer=3,
                 block_size=20,
                 dropout=0.1,
                 kp_activation='tanh',
                 predict_delta=False,
                 max_delta=1.5,
                 positional_bias=False,
                 max_particles=None,
                 context_dim=7,
                 attn_norm_type='rms',
                 n_fg_particles=None,
                 ctx_pool_mode='none',
                 ctx_mode='adaln',
                 particle_score=False,
                 particle_positional_embed=True,
                 scale_anchor=None,
                 init_std=0.02,
                 pint_ctx_layers=6,
                 pint_ctx_heads=8,
                 ctx_dist='gauss',
                 n_ctx_categories=4,
                 n_ctx_classes=4,
                 residual_modulation=True,
                 context_gate=True,
                 context_decoder=None,
                 features_dist='gauss',
                 n_fg_categories=8,
                 n_fg_classes=4,
                 n_bg_categories=4,
                 n_bg_classes=4,
                 particle_anchors=None,
                 scale_init=None,
                 obj_on_min=1e-4,
                 obj_on_max=100,
                 use_z_orig=True,
                 n_views=1,
                 action_condition=False,
                 action_dim=0,
                 random_action_condition=False,
                 random_action_dim=0,
                 null_action_embed=False,
                 pos_embed_t_adaln=True,
                 pos_embed_p_adaln=True,
                 pos_embed_objon_adaln=False,
                 # GNN-Mamba3 specific
                 gnn_hidden_dim=128,
                 gnn_context_dim=128,
                 gnn_layers=3,
                 egnn_n_rbf=16,
                 mamba_d_state=64,
                 mamba_headdim=64,
                 mamba_d_conv=4,
                 mamba_expand=2,
                 mamba_is_mimo=True,
                 mamba_mimo_rank=4,
                 mamba_chunk_size=16,
                 ):
        super().__init__()

        self.predict_delta = predict_delta
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.max_delta = max_delta
        self.max_particles = max_particles
        self.n_fg_particles = n_fg_particles
        self.learned_feature_dim = features_dim
        self.learned_bg_feature_dim = bg_features_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        self.context_dist = ctx_dist
        self.n_ctx_categories = n_ctx_categories
        self.n_ctx_classes = n_ctx_classes
        self.context_dim = context_dim
        self.particle_score = particle_score
        self.attn_norm_type = attn_norm_type
        assert ctx_mode in ['add', 'cat', 'token', 'film', 'adaln']
        self.ctx_mode = ctx_mode
        self.ctx_pool_mode = ctx_pool_mode
        self.init_std = init_std
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_z_orig = use_z_orig
        self.n_views = n_views

        # Actions
        self.action_condition = action_condition
        self.action_dim = action_dim
        self.random_action_condition = random_action_condition
        self.random_action_dim = random_action_dim
        self.learn_null_action_embed = null_action_embed

        # Token adaln
        self.pos_embed_t_adaln = pos_embed_t_adaln
        self.pos_embed_p_adaln = pos_embed_p_adaln
        self.pos_embed_objon_adaln = pos_embed_objon_adaln

        if self.learn_null_action_embed and self.action_condition:
            self.null_action_embeddings = nn.Parameter(
                self.init_std * torch.randn(1, 1, self.action_dim))
        else:
            self.null_action_embeddings = None

        if scale_anchor is None:
            self.register_buffer('scale_anchor', torch.tensor(0.0))
        else:
            self.register_buffer('scale_anchor',
                                 torch.tensor(np.log(0.75 * scale_anchor / (1 - 0.75 * scale_anchor + 1e-5))))
        if particle_anchors is None:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, self.n_fg_particles))
            self.use_z_orig = False
        else:
            self.register_buffer('particles_anchor', particle_anchors)

        self.particle_pos_embed = particle_positional_embed and not self.pos_embed_p_adaln

        # Lazy import LPWM modules
        ParticleFeatureProjection, ParticleFeatureDecoderDyn, RMSNorm = _get_lpwm_imports()

        # Particle projection (reused from LPWM)
        proj_max_particles = self.n_fg_particles
        self.particle_projection = ParticleFeatureProjection(
            features_dim, bg_features_dim,
            hidden_dim, self.projection_dim, context_dim=context_dim,
            max_particles=proj_max_particles, add_embedding=True,
            ctx_cond_mode=self.ctx_mode,
            particle_positional_embed=self.particle_pos_embed,
            init_std=self.init_std, particle_score=self.particle_score,
            norm_layer=True, use_z_orig=self.use_z_orig)

        # Context projection for AdaLN
        if self.ctx_mode == 'adaln' and self.context_dim > 0:
            self.context_proj = nn.Linear(self.context_dim, hidden_dim)
            if self.action_condition and self.action_dim > 0:
                self.action_proj = nn.Linear(self.action_dim, hidden_dim)
            else:
                self.action_proj = None
            if self.random_action_condition and self.random_action_dim > 0:
                self.random_action_proj = nn.Linear(self.random_action_dim, hidden_dim)
            else:
                self.random_action_proj = None
            self.cond_activation = nn.GELU()
        else:
            self.context_proj = None
            self.action_proj = None
            self.cond_activation = None

        if self.n_views > 1:
            self.view_embeddings = nn.Parameter(
                self.init_std * torch.randn(1, self.n_views, 1, 1, self.projection_dim))
        else:
            self.view_embeddings = None

        if self.pos_embed_p_adaln and (self.ctx_mode == 'adaln'):
            n_particles = self.n_views * (self.n_fg_particles + 1)
            self.pos_p_embeddings = nn.Parameter(
                self.init_std * torch.randn(1, n_particles, 1, hidden_dim))
        if self.pos_embed_objon_adaln:
            self.objon_embeddings = nn.Sequential(
                nn.Linear(1, hidden_dim), RMSNorm(hidden_dim), nn.GELU())

        # === CORE REPLACEMENT: GNNMamba3Transformer instead of ParticleSpatioTemporalTransformer ===
        self.particle_transformer = GNNMamba3Transformer(
            n_embed=self.projection_dim,
            n_layer=n_layer,
            block_size=block_size,
            output_dim=self.projection_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_context_dim=gnn_context_dim,
            gnn_layers=gnn_layers,
            egnn_n_rbf=egnn_n_rbf,
            mamba_d_state=mamba_d_state,
            mamba_headdim=mamba_headdim,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_is_mimo=mamba_is_mimo,
            mamba_mimo_rank=mamba_mimo_rank,
            mamba_chunk_size=mamba_chunk_size,
            context_cond=(self.ctx_mode == 'adaln'),
            init_std=self.init_std,
            pos_embed_t_adaln=self.pos_embed_t_adaln,
        )

        # Particle decoder (reused from LPWM)
        self.particle_decoder = ParticleFeatureDecoderDyn(
            self.projection_dim, features_dim, bg_features_dim,
            hidden_dim, kp_activation=kp_activation, max_delta=max_delta,
            context_dim=context_dim, ctx_as_token=(self.ctx_mode == 'token'),
            dec_ctx=False, norm_type=attn_norm_type, dropout=dropout,
            particle_score=self.particle_score, features_dist=self.features_dist,
            n_fg_categories=n_fg_categories, n_fg_classes=n_fg_classes,
            n_bg_categories=n_bg_categories, n_bg_classes=n_bg_classes,
            scale_init=scale_init)

        self.context_decoder = context_decoder

    def get_block_size(self):
        return self.particle_transformer.get_block_size()

    def init_weights(self):
        self.particle_projection.init_weights()
        self.particle_decoder.init_weights()

    def _prepare_context_conditioning(self, particle_proj_int, z_context_v, bs, timestep_horizon,
                                      actions=None, actions_mask=None, z_obj_on=None):
        """Prepare AdaLN context conditioning (same logic as DLPDynamics)."""
        if self.ctx_mode != 'adaln':
            return None

        if self.random_action_condition:
            random_actions = torch.rand(
                particle_proj_int.shape[0], particle_proj_int.shape[2],
                self.random_action_dim, device=particle_proj_int.device)
            c_random_action = self.random_action_proj(random_actions)
            if len(c_random_action.shape) == 3:
                c_random_action = c_random_action.unsqueeze(1).repeat(
                    1, particle_proj_int.shape[1], 1, 1)
        else:
            c_random_action = 0

        if self.action_condition and actions is not None:
            if self.learn_null_action_embed and actions_mask is not None:
                if len(actions_mask.shape) == 2:
                    actions_mask = actions_mask.bool().unsqueeze(-1)
                null_action_embeds = self.null_action_embeddings.expand(
                    actions.size(0), actions.size(1), -1)
                actions = actions * actions_mask + null_action_embeds * (~actions_mask)
            c_action = self.action_proj(actions)
            if len(c_action.shape) == 3:
                c_action = c_action.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1, 1)
        else:
            c_action = 0

        c = self.context_proj(z_context_v)
        c = c.reshape(bs, timestep_horizon, *c.shape[1:])
        if len(c.shape) == 3:
            c = c.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1, 1)
        elif c.shape[2] != particle_proj_int.shape[1]:
            c = c.permute(0, 2, 1, 3)
            c = c.repeat(1, particle_proj_int.shape[1], 1, 1)
        else:
            c = c.permute(0, 2, 1, 3)
        c = c + c_action + c_random_action
        c = self.cond_activation(c)

        # Positional embedding for particles
        if self.pos_embed_p_adaln:
            c_pe = self.pos_p_embeddings.repeat(c.shape[0], 1, c.shape[2], 1)
            c = c + c_pe

        # obj_on embedding
        if self.pos_embed_objon_adaln and z_obj_on is not None:
            c_objon = self.objon_embeddings(z_obj_on)
            c_objon_bg = torch.zeros(
                c_objon.shape[0], c_objon.shape[1], 1, c_objon.shape[-1],
                device=c_objon.device)
            c_objon = torch.cat([c_objon, c_objon_bg], dim=2)
            c_objon = c_objon.permute(0, 2, 1, 3)
            if self.n_views > 1:
                c_objon = c_objon.reshape(
                    -1, self.n_views * c_objon.shape[1], c_objon.shape[2], c_objon.shape[-1])
            c = c + c_objon

        return c

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features,
                z_context, z_score=None, actions=None, actions_mask=None):
        """Forward dynamics prediction (same interface as DLPDynamics.forward).

        Args:
            z: [bs, T, n_particles, 2] positions
            z_scale: [bs, T, n_particles, 2]
            z_obj_on: [bs, T, n_particles, 1]
            z_depth: [bs, T, n_particles, 1]
            z_features: [bs, T, n_particles, feat_dim]
            z_bg_features: [bs, T, bg_feat_dim]
            z_context: [bs, T, context_dim]
            z_score: [bs, T, n_particles, 1] optional
            actions: [bs, T, action_dim] optional
            actions_mask: [bs, T] optional

        Returns:
            dict with keys: mu, logvar, mu_features, logvar_features,
                           obj_on_a, obj_on_b, mu_depth, logvar_depth,
                           mu_scale, logvar_scale, mu_bg_features, logvar_bg_features,
                           mu_context, logvar_context, mu_score, logvar_score
        """
        bs, timestep_horizon, n_particles, _ = z.shape

        mu_context = logvar_context = None

        # Project particles
        z_v = z.reshape(bs * timestep_horizon, *z.shape[2:])
        z_scale_v = z_scale.reshape(bs * timestep_horizon, *z_scale.shape[2:])
        z_obj_on_v = z_obj_on.reshape(bs * timestep_horizon, *z_obj_on.shape[2:])
        z_depth_v = z_depth.reshape(bs * timestep_horizon, *z_depth.shape[2:])
        z_features_v = z_features.reshape(bs * timestep_horizon, *z_features.shape[2:])
        z_bg_features_v = z_bg_features.reshape(bs * timestep_horizon, *z_bg_features.shape[2:])
        z_context_v = z_context.reshape(bs * timestep_horizon, *z_context.shape[2:])
        if self.use_z_orig:
            z_orig_v = self.particles_anchor.repeat(bs * timestep_horizon, 1, 1)
        else:
            z_orig_v = None
        if z_score is not None:
            z_score_v = z_score.reshape(bs * timestep_horizon, *z_score.shape[2:])
        else:
            z_score_v = z_score

        particle_projection = self.particle_projection(
            z_v, z_scale_v, z_obj_on_v, z_depth_v, z_features_v,
            z_bg_features_v, z_context_v, z_score_v, z_orig_v)
        # [bs * T, n_particles + 1, projection_dim]

        particle_proj_int = particle_projection.view(
            bs, timestep_horizon, *particle_projection.shape[1:])
        # [bs, T, n_particles + 1, projection_dim]
        particle_proj_int = particle_proj_int.permute(0, 2, 1, 3)
        # [bs, n_particles + 1, T, projection_dim]

        # Context conditioning
        c = self._prepare_context_conditioning(
            particle_proj_int, z_context_v, bs, timestep_horizon,
            actions=actions, actions_mask=actions_mask, z_obj_on=z_obj_on)

        # Construct raw 3D positions for GNN distance computation
        # z: [bs, T, n_particles, 2], z_depth: [bs, T, n_particles, 1]
        pos_3d = torch.cat([z, z_depth], dim=-1)  # [bs, T, n_particles, 3]
        # Add bg dummy position (zeros)
        bg_pos = torch.zeros(bs, timestep_horizon, 1, 3, device=z.device)
        pos_3d = torch.cat([pos_3d, bg_pos], dim=2)  # [bs, T, n_particles+1, 3]
        pos_3d = pos_3d.permute(0, 2, 1, 3)  # [bs, N+1, T, 3]

        # Multi-view handling
        if self.n_views > 1:
            particle_proj_int = particle_proj_int.view(
                -1, self.n_views, particle_proj_int.shape[1], *particle_proj_int.shape[2:])
            particle_proj_int = particle_proj_int + self.view_embeddings
            particle_proj_int = particle_proj_int.reshape(
                particle_proj_int.shape[0], -1, *particle_proj_int.shape[3:])
            pos_3d = pos_3d.view(-1, self.n_views, pos_3d.shape[1], *pos_3d.shape[2:])
            pos_3d = pos_3d.reshape(pos_3d.shape[0], -1, *pos_3d.shape[3:])
            if c is not None:
                c = c.reshape(-1, self.n_views * c.shape[1], *c.shape[2:])

        # === Forward through GNN-Mamba3 Transformer ===
        particles_trans = self.particle_transformer(particle_proj_int, pos=pos_3d, c=c)
        # [bs, n_particles + 1, T, projection_dim]

        if self.n_views > 1:
            particles_trans = particles_trans.reshape(bs, -1, *particles_trans.shape[2:])

        particles_trans = particles_trans.permute(0, 2, 1, 3)
        # [bs, T, n_particles + 1, projection_dim]

        # Decode
        particles_trans = particles_trans.reshape(-1, *particles_trans.shape[2:])
        # [bs * T, n_particles + 1, projection_dim]
        particle_decoder_out = self.particle_decoder(particles_trans)

        mu = particle_decoder_out['mu_offset']
        logvar = particle_decoder_out['logvar_offset']

        obj_on_a_gate = particle_decoder_out['lobj_on_a'].sigmoid()
        obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
        obj_on_b_gate = 1 - (
            particle_decoder_out['lobj_on_b'] * 0 + particle_decoder_out['lobj_on_a']).sigmoid()
        obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()

        mu_depth = particle_decoder_out['mu_depth']
        logvar_depth = particle_decoder_out['logvar_depth']
        mu_scale = particle_decoder_out['mu_scale']
        logvar_scale = particle_decoder_out['logvar_scale']
        mu_features = particle_decoder_out['mu_features']
        logvar_features = particle_decoder_out['logvar_features']
        mu_bg_features = particle_decoder_out['mu_bg_features']
        logvar_bg_features = particle_decoder_out['logvar_bg_features']
        mu_score = particle_decoder_out['mu_score']
        logvar_score = particle_decoder_out['logvar_score']

        mu_scale = mu_scale + self.scale_anchor
        if self.use_z_orig:
            mu = self.particles_anchor + mu

        if self.predict_delta:
            mu = z_v + mu

        # Reshape to [bs, T, ...]
        mu = mu.view(bs, timestep_horizon, *mu.shape[1:])
        logvar = logvar.view(bs, timestep_horizon, *logvar.shape[1:])
        obj_on_a = obj_on_a.view(bs, timestep_horizon, *obj_on_a.shape[1:])
        obj_on_b = obj_on_b.view(bs, timestep_horizon, *obj_on_b.shape[1:])
        mu_depth = mu_depth.view(bs, timestep_horizon, *mu_depth.shape[1:])
        logvar_depth = logvar_depth.view(bs, timestep_horizon, *logvar_depth.shape[1:])
        mu_scale = mu_scale.view(bs, timestep_horizon, *mu_scale.shape[1:])
        logvar_scale = logvar_scale.view(bs, timestep_horizon, *logvar_scale.shape[1:])
        mu_features = mu_features.view(bs, timestep_horizon, *mu_features.shape[1:])
        logvar_features = logvar_features.view(bs, timestep_horizon, *logvar_features.shape[1:])
        mu_bg_features = mu_bg_features.view(bs, timestep_horizon, *mu_bg_features.shape[1:])
        logvar_bg_features = logvar_bg_features.view(bs, timestep_horizon, *logvar_bg_features.shape[1:])
        if self.particle_score and mu_score is not None:
            mu_score = mu_score.view(bs, timestep_horizon, *mu_score.shape[1:])
            logvar_score = logvar_score.view(bs, timestep_horizon, *logvar_score.shape[1:])

        output_dict = {
            'mu': mu, 'logvar': logvar,
            'mu_features': mu_features, 'logvar_features': logvar_features,
            'obj_on_a': obj_on_a.squeeze(-1), 'obj_on_b': obj_on_b.squeeze(-1),
            'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
            'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
            'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features,
            'mu_context': mu_context, 'logvar_context': logvar_context,
            'mu_score': mu_score, 'logvar_score': logvar_score,
        }
        return output_dict

    def sample(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features,
               z_context=None, z_score=None, steps=10, deterministic=False,
               deterministic_particles=True, actions=None, actions_mask=None,
               lang_embed=None, z_goal=None, return_context_posterior=False):
        """Autoregressive sampling (same interface as DLPDynamics.sample).

        Iteratively predicts next-step particle states.
        """
        block_size = self.particle_transformer.get_block_size()

        if z_score is None:
            z_score = torch.zeros(z.shape[0], z.shape[1], z.shape[2], 1,
                                  dtype=torch.float, device=z.device)

        mu_context_posterior = z_context_posterior = z_context
        bs, timestep_horizon, n_particles, _ = z.shape

        for k in range(steps):
            # Context generation (if context_decoder exists)
            if self.context_dim > 0 and self.context_decoder is not None:
                start_step = max(z.shape[1] - block_size, 0)
                end_step = min(start_step + block_size, z.shape[1])
                if z_context is None or z_context.shape[1] < z.shape[1]:
                    if actions is not None:
                        actions_in = actions[:, start_step:end_step]
                    else:
                        actions_in = None
                    if actions_mask is not None:
                        actions_mask_in = actions_mask[:, start_step:end_step]
                    else:
                        actions_mask_in = None
                    ctx_dec_out = self.context_decoder(
                        z=z[:, -block_size:], z_scale=z_scale[:, -block_size:],
                        z_obj_on=z_obj_on[:, -block_size:], z_depth=z_depth[:, -block_size:],
                        z_features=z_features[:, -block_size:],
                        z_bg_features=z_bg_features[:, -block_size:],
                        z_score=z_score[:, -block_size:],
                        deterministic=deterministic, encode_posterior=return_context_posterior,
                        encode_prior=True, actions=actions_in, actions_mask=actions_mask_in,
                        lang_embed=lang_embed, z_goal=z_goal)
                    z_context_last = ctx_dec_out['z_context_dyn'][:, -1:]
                    if z_context is None:
                        z_context = z_context_last
                    else:
                        z_context = torch.cat([z_context, z_context_last], dim=1)

            # Prepare inputs for dynamics
            start_step = max(z.shape[1] - block_size, 0)
            actual_t = min(block_size, z.shape[1])
            if self.context_dim > 0 and z_context is not None:
                # Context may have fewer timesteps than z if no context_decoder
                ctx_end = min(z_context.shape[1], start_step + actual_t)
                z_context_slice = z_context[:, start_step:ctx_end]
                # Pad context to match particle timesteps if needed
                if z_context_slice.shape[1] < actual_t:
                    pad_len = actual_t - z_context_slice.shape[1]
                    last_ctx = z_context_slice[:, -1:].expand(-1, pad_len, -1)
                    z_context_slice = torch.cat([z_context_slice, last_ctx], dim=1)
                z_context_v = z_context_slice.reshape(-1, *z_context_slice.shape[2:])
            else:
                z_context_v = None

            z_v = z[:, -block_size:].reshape(-1, *z.shape[2:])
            z_scale_v = z_scale[:, -block_size:].reshape(-1, *z_scale.shape[2:])
            z_obj_on_v = z_obj_on[:, -block_size:].reshape(-1, *z_obj_on.shape[2:])
            z_depth_v = z_depth[:, -block_size:].reshape(-1, *z_depth.shape[2:])
            z_features_v = z_features[:, -block_size:].reshape(-1, *z_features.shape[2:])
            z_bg_features_v = z_bg_features[:, -block_size:].reshape(-1, *z_bg_features.shape[2:])
            z_score_v = z_score[:, -block_size:].reshape(-1, *z_score.shape[2:])
            if self.use_z_orig:
                z_orig_v = self.particles_anchor.repeat(z_v.shape[0], 1, 1)
            else:
                z_orig_v = None

            # Project particles
            particle_projection = self.particle_projection(
                z_v, z_scale_v, z_obj_on_v, z_depth_v, z_features_v,
                z_bg_features_v, z_context_v, z_score_v, z_orig_v)

            particle_proj_int = particle_projection.view(bs, -1, *particle_projection.shape[1:])
            particle_proj_int = particle_proj_int.permute(0, 2, 1, 3)

            # Context conditioning
            if self.ctx_mode == 'adaln' and z_context_v is not None:
                c = self.context_proj(z_context_v)
                c = c.reshape(bs, -1, *c.shape[1:])
                if len(c.shape) == 3:
                    c = c.unsqueeze(1)
                elif c.shape[2] != particle_proj_int.shape[1]:
                    c = c.permute(0, 2, 1, 3)
                    c = c.repeat(1, particle_proj_int.shape[1], 1, 1)
                else:
                    c = c.permute(0, 2, 1, 3)
                c = self.cond_activation(c)
            else:
                c = None

            # Construct raw 3D positions for GNN distance computation
            z_slice = z[:, -block_size:]
            z_depth_slice = z_depth[:, -block_size:]
            pos_3d = torch.cat([z_slice, z_depth_slice], dim=-1)  # [bs, T', n_particles, 3]
            bg_pos = torch.zeros(bs, pos_3d.shape[1], 1, 3, device=z.device)
            pos_3d = torch.cat([pos_3d, bg_pos], dim=2)  # [bs, T', n_particles+1, 3]
            pos_3d = pos_3d.permute(0, 2, 1, 3)  # [bs, N+1, T', 3]

            if self.n_views > 1:
                particle_proj_int = particle_proj_int.view(
                    -1, self.n_views, particle_proj_int.shape[1], *particle_proj_int.shape[2:])
                particle_proj_int = particle_proj_int + self.view_embeddings
                particle_proj_int = particle_proj_int.reshape(
                    particle_proj_int.shape[0], -1, *particle_proj_int.shape[3:])
                pos_3d = pos_3d.view(-1, self.n_views, pos_3d.shape[1], *pos_3d.shape[2:])
                pos_3d = pos_3d.reshape(pos_3d.shape[0], -1, *pos_3d.shape[3:])
                if c is not None:
                    c = c.reshape(-1, self.n_views * c.shape[1], *c.shape[2:])

            # Forward transformer
            particles_trans = self.particle_transformer(particle_proj_int, pos=pos_3d, c=c)
            if self.n_views > 1:
                particles_trans = particles_trans.reshape(bs, -1, *particles_trans.shape[2:])

            # Take last timestep prediction
            particles_trans = particles_trans[:, :, -1]  # [bs, N+1, D]

            # Decode
            particle_decoder_out = self.particle_decoder(particles_trans)
            mu = particle_decoder_out['mu_offset']
            logvar = particle_decoder_out['logvar_offset']

            obj_on_a_gate = particle_decoder_out['lobj_on_a'].sigmoid()
            obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
            obj_on_b_gate = 1 - (
                particle_decoder_out['lobj_on_b'] * 0 + particle_decoder_out['lobj_on_a']).sigmoid()
            obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()

            mu_depth = particle_decoder_out['mu_depth']
            logvar_depth = particle_decoder_out['logvar_depth']
            mu_scale = particle_decoder_out['mu_scale']
            logvar_scale = particle_decoder_out['logvar_scale']
            mu_features = particle_decoder_out['mu_features']
            logvar_features = particle_decoder_out['logvar_features']
            mu_bg_features = particle_decoder_out['mu_bg_features']
            logvar_bg_features = particle_decoder_out['logvar_bg_features']

            # Reshape
            mu = mu.view(bs, 1, *mu.shape[1:])
            logvar = logvar.view(bs, 1, *logvar.shape[1:])
            obj_on_a = obj_on_a.view(bs, 1, *obj_on_a.shape[1:])
            obj_on_b = obj_on_b.view(bs, 1, *obj_on_b.shape[1:])
            mu_depth = mu_depth.view(bs, 1, *mu_depth.shape[1:])
            mu_scale = mu_scale.view(bs, 1, *mu_scale.shape[1:])
            mu_features = mu_features.view(bs, 1, *mu_features.shape[1:])
            mu_bg_features = mu_bg_features.view(bs, 1, *mu_bg_features.shape[1:])

            mu_scale = mu_scale + self.scale_anchor
            if self.use_z_orig:
                mu = self.particles_anchor.unsqueeze(1) + mu
            if self.predict_delta:
                mu = z[:, -1].unsqueeze(1) + mu

            beta_dist = Beta(obj_on_a, obj_on_b)

            if deterministic or deterministic_particles:
                new_z = mu
                new_z_depth = mu_depth
                new_z_scale = mu_scale
                new_z_features = mu_features
                new_z_bg_features = mu_bg_features
                new_z_obj_on = beta_dist.mean
                new_z_score = logvar.sum(-1, keepdim=True).view(bs, 1, *logvar.shape[2:-1], 1)
            else:
                new_z = reparameterize(mu, logvar)
                new_z_depth = reparameterize(mu_depth, logvar_depth.view(bs, 1, *logvar_depth.shape[1:]))
                new_z_scale = reparameterize(mu_scale, logvar_scale.view(bs, 1, *logvar_scale.shape[1:]))
                new_z_features = reparameterize(mu_features, logvar_features.view(bs, 1, *logvar_features.shape[1:]))
                new_z_bg_features = reparameterize(mu_bg_features, logvar_bg_features.view(bs, 1, *logvar_bg_features.shape[1:]))
                new_z_obj_on = beta_dist.sample()
                new_z_score = logvar.sum(-1, keepdim=True).view(bs, 1, *logvar.shape[2:-1], 1)

            z = torch.cat([z, new_z], dim=1)
            z_depth = torch.cat([z_depth, new_z_depth], dim=1)
            z_scale = torch.cat([z_scale, new_z_scale], dim=1)
            z_features = torch.cat([z_features, new_z_features], dim=1)
            z_bg_features = torch.cat([z_bg_features, new_z_bg_features], dim=1)
            z_obj_on = torch.cat([z_obj_on, new_z_obj_on], dim=1)
            z_score = torch.cat([z_score, new_z_score], dim=1)

        out_dict = {
            'z': z, 'z_scale': z_scale, 'z_obj_on': z_obj_on, 'z_depth': z_depth,
            'z_features': z_features, 'z_bg_features': z_bg_features,
            'z_context': z_context, 'z_score': z_score,
            'z_context_posterior': z_context_posterior,
            'mu_context_posterior': mu_context_posterior,
        }
        return out_dict
