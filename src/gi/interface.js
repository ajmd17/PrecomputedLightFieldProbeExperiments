/**
 * IIndirectLightingSystem — modular interface for GI backends.
 *
 * Lifecycle:
 *   init(gl, sceneData)       one-time setup
 *   precompute(gl, camera)    optional bake step (e.g. cubemap rendering)
 *   bindUniforms(gl, program) upload per-frame data before the GI pass
 *   destroy(gl)               tear down GPU resources
 *
 * sceneData shape:
 *   { sceneVAOs, M, bounds, voxels, ... }
 */
export class IndirectLightingSystem {
  init(gl, sceneData) {
    throw new Error('IndirectLightingSystem.init() not implemented');
  }
  precompute(gl) {
    // optional — default no-op
  }
  bindUniforms(gl, program, camera) {
    throw new Error('IndirectLightingSystem.bindUniforms() not implemented');
  }
  destroy(gl) {
    // optional
  }
}
