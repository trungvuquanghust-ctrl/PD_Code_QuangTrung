# PECT architecture boundary

TIM_2026 exposes a single architecture: PECT. Its canonical path combines four
elements that must be considered together:

1. ResNet12 produces a dense spatial feature map instead of only a global
   embedding.
2. A learned local-token projector creates the descriptors used by transport.
3. Unbalanced optimal transport permits the transported mass to adapt when
   local regions do not match reliably; the local logit uses threshold-mass
   scoring rather than plain prototype distance.
4. A global prototype score is added as a residual with fixed weight `0.1`.

The standalone contribution boundary is therefore not "UOT alone" and not
"global prototypes alone". It is the episode-level combination of local
mass-aware transport evidence with a deliberately small global correction.

The included ablations test that boundary directly:

- `pect_no_global` and `pect_global_only` isolate the two score branches.
- the global-weight sweep tests whether `0.1` is a meaningful residual scale.
- the rho/latent-rho variants test fixed versus episode-adaptive transport mass.
- full OT and partial OT change the transport family while keeping the rest of
  the protocol fixed.
- class-pooled removes shot-decomposed local matching before transport.
- cost-only removes the transported-mass reward from the local score.

This is a code-level architecture comparison, not a claim that every component
is individually novel relative to all published literature. Any publication
novelty claim should be supported by a separate, current literature review.
