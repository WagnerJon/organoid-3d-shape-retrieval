# Classical and fractional 3D Zernike descriptors

Independent research implementation of Ashton & Sit-Seth (2026), [New 3D Fractional-Order Zernike Moments for Shape Analysis](https://doi.org/10.3390/appliedmath6100174). The existing classical reference uses the [ZMPY3D author implementation](https://github.com/tawssie/ZMPY3D).

`scripts/zernike_fractional_core.py` computes rotationally invariant norms of complex 3D Zernike moments. Order20 gives121 entries indexed by `(n,l)`, with `0≤l≤n≤20` and even `n−l`. The norm over `m=−l,…,l` removes orientation in the continuous construction. Finite sampling still introduces rotation error. Translation and physical size are removed before integration; size remains available in the original morphometry. Phase and chirality are discarded, so these are not complete invariants and do not uniquely determine a reconstruction.

## Radial basis and a formula discrepancy

For `k=(n−l)/2`, the implementation evaluates

\[
R_{n,l;\alpha}(r)=\sqrt{\frac{2\alpha n+3}{3}}\,
r^{\alpha l}P_k^{(0,\,l+3/(2\alpha)-1)}(2r^{2\alpha}-1).
\]

Here `P` is the Jacobi polynomial, evaluated in float64 using SciPy. Substitution `t=r^(2α)` gives Jacobi weight `t^(l+3/(2α)−1)`, directly establishing the paper's Eq.4:

\[
3\int_0^1 R_{n,l;\alpha}(r)R_{n',l;\alpha}(r)r^2dr=\delta_{nn'}.
\]

This is the paper's radial family, evaluated without summing large alternating monomials. It agrees with the Section4 recurrence and reduces to classical 3D Zernike at `α=1`. The spherical harmonics use the paper's angular mean-square normalization; SciPy's unit-integral harmonics are converted accordingly.

**The printed Eq.9 needs care.** It defines a unit-step rising factorial of `(2αl+2αν+3)` divided by `(2α)^k`. Read literally, this disagrees with Eq.4 and the Section4 recurrence. The consistent coefficient uses `(l+ν+3/(2α))_k` (or an equivalent product with step `2α` before division). Even at `α=1,n=4,l=0`, the literal printed formula yields `sqrt(11/3)*(1.5−7.5r²+7r⁴)`, whereas the orthogonal polynomial is `sqrt(11/3)*(1.875−8.75r²+7.875r⁴)`. The former has a nonzero inner product `sqrt(77)/63≈0.1393` with the normalized `n=2,l=0` basis. We follow the orthogonality condition and recurrence. This is a mathematical implementation choice, not a claim of reproducing the authors' unavailable implementation.

## Controlled five-organoid experiment

`scripts/run_fractional_zernike.py` reuses the fixed100-query cross-organoid plan from `results/five_organoid_shape_retrieval`. It encodes all961 objects and preserves the747-object QC/main-organoid retrieval gallery. Query neighbors come from the other four organoids. Original masks, meshes, measurements, exclusions, Atlas dataset files and recommendations are never modified.

- Primary: filled-volume input, `α=0.38` against classical `α=1`, with identical L2 normalization and Euclidean distance.
- Prespecified secondary exponents: `α=0.14` and `0.20`, taken from the paper, not selected by optimizing these cells.
- Input control: binary surface-distance band, half-width `sqrt(3)/64` normalized units, compared with filled-volume occupancy. This corresponds to a half-voxel diagonal for Cartesian spacing `2/64`.
- Both inputs share an exact volume centroid for closed, consistently oriented meshes and an area-centroid fallback for open surfaces. The furthest vertex maps to radius0.9. The paper instead centers its surface voxels. This shared normalization isolates the radial-basis comparison.
- Filled volume uses logistic signed-distance occupancy with width`1/64`, matching the existing ZMPY3D experiment's input convention.
- Integration uses96 radial Gauss–Legendre nodes,32 polar nodes and64 azimuthal nodes. Signed-distance queries run in chunks of at most32,768 points; whole microscopy stacks are not loaded. Two CPU processes limit memory.
- Secondary paper-inspired scaling uses mass-normalized invariant vectors and per-coordinate MinMax fitted on included objects from the other four organoids only. Constant coordinates are suppressed; queries are not clipped. The paper fits its scaler on the entire benchmark. These differences preclude claiming reproduction of its reported recognition numbers.

Evaluation uses the existing768-point,24-start proper-rotation surface-ICP cost, fixed16 reference cells per query, and25 prespecified rotation/noise probes. Noise is a smooth normal displacement with2% RMS amplitude relative to normalized surface radius. These are geometric diagnostics, not verified biological labels. Intervals bootstrap whole organoids, with only five available; all reported uncertainty is descriptive.

Additional diagnostics encode all72 demo objects and check a finer160×48×96 quadrature on one fixed probe per organoid. Synthetic family recognition excludes the query's own gallery. The original classical ZMPY3D vectors provide an independent numerical integration crosscheck.

## Running

Requires the existing five-organoid mesh dataset and benchmark caches. Use the project environment:

```sh
.venv/bin/python -m unittest discover -s tests -p test_zernike_fractional.py -v
.venv/bin/python scripts/run_fractional_zernike.py encode --workers 2
.venv/bin/python scripts/run_fractional_zernike.py evaluate
.venv/bin/python scripts/run_fractional_zernike.py probes --workers 2
.venv/bin/python scripts/run_fractional_zernike.py audit
.venv/bin/python scripts/run_fractional_zernike.py demo
.venv/bin/python scripts/report_fractional_zernike.py
```

The encoder supports direct use on any mesh through `mesh_descriptors(vertices, faces)`. The experiment runner is deliberately tied to the existing, fixed five-organoid benchmark rather than silently creating a new evaluation split.

Outputs are under `results/fractional_zernike/`: the complete report, comparison figure, source/config hashes,121D vectors, per-query scores, perturbation probes, convergence checks and cross-organoid neighbor lists. These are research artifacts, not bundled Atlas methods; large benchmark outputs are excluded from source releases.

## Observed results (2026-10-23)

All 961 biological objects and 72 synthetic demo objects were encoded successfully. The fixed 100-query comparison gave the following mean cross-organoid neighbor costs (lower is better):

| Descriptor | Geometry cost | Identity after 2% surface noise |
|---|---:|---:|
| Existing raw multi-shell harmonics | 0.04847 | 23/25 |
| Existing classical ZMPY3D | 0.05246 | 21/25 |
| New classical Zernike, filled volume | 0.05258 | 23/25 |
| Fractional α=0.38, filled volume | 0.05237 | 22/25 |
| Classical Zernike, surface band | 0.05004 | 25/25 |
| Fractional α=0.38, surface band | 0.05032 | 25/25 |
| Exploratory fractional α=0.14, volume + fold MinMax | 0.04866 | 21/25 |

Changing only the volume exponent to α=0.38 reduced cost by **0.4%**, with a paired organoid-bootstrap 95% interval of **−1.8% to +3.2%**. This does not establish an improvement. The best exploratory fractional row was close to existing shells but had weaker noise identity retrieval. Surface input improved noise stability in this small probe set; it did not improve near/far ordering over shells. Keep raw multi-shell harmonics as the current practical recommendation.

All eight new variants recovered the correct rotated cell in 25/25 probes and the correct synthetic family in 72/72 cases. All finer-quadrature probes retained identity (5/5 per variant), but surface-band integration had appreciably more sampling drift than filled volume. These checks support numerical usability, not exact invariance or exhaustive convergence.

Two-worker encoding took 157 seconds, with a median 0.29 seconds per cell and a maximum recorded 0.68 GiB RSS per worker on this machine. Five unit checks passed. Full per-query values, paired intervals, numerical crosschecks and all exploratory variants are in the generated report.
