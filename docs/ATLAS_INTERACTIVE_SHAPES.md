# Full-screen plots and shape discovery

The [full-data extension](ATLAS_FULL_DATA_SHAPES.md) expands the regular comparison
methods to all cells. The 48-cell exact CAJAL reference is available through the
separate validation scope, so it does not restrict the all-method overview.

Every plot in **Explore** and **Compare Methods** has two buttons:

- The expansion icon opens that plot in its own native macOS full-screen window.
- **3 PCs** opens a rotatable plot of the first three principal components.
  Distance-based methods instead have a **3D** button and display three MDS axes.
  CAJAL's packed distance rows and spatial PCA's sign frames are never treated as
  ordinary embedding coordinates.

In the plot window, switch between two and three axes, reset the camera, or toggle
macOS full screen to use an ordinary resizable window. Drag to orbit, scroll to
zoom, and click a point to mark/select the same object in Atlas's tables. Two-axis
mode retains rectangular brushing. **Inspect selection in 3D** returns to the
original reconstruction explorer. PCA and MDS use all included objects with that
method's results, matching the existing descriptive fitting scope; displayed
objects continue to respect the current filters and the opening comparison cohort.
The three spatial plot axes share a scale, with red/green/blue corresponding to
PC1/PC2/PC3 (or MDS axis 1/2/3). Variance/inertia is shown per axis. Missing or
degenerate dimensions are padded with zero, not fabricated.

## Discover Shape Groups tab

**Similar shapes by method** shows the selected query and each method's closest
matches in full feature space. The first two matches have interactive original
surface previews; three more matches are available as comparison buttons.
Rotation/zoom can be linked, and previews share a physical camera scale across
methods. **Use as query**, **Compare**, and **Mark nearest 5** connect these shapes
back to the explorer, comparison sheet and linked tables. A comparison sheet
opened from a method's gallery reports that method's distance. **Other stacks
only** avoids returning cells from the query's image stack.

**Discover groups** runs HDBSCAN on a selected method's complete comparison
distance matrix for the visible cohort. It does not cluster the 2D/3D plot.
There is no fixed group count; a single group, multiple groups, or no groups are
valid results. Minimum group size and density-neighbor controls change the
definition of a dense group. Unassigned objects remain available for review.
**Balanced groups** (default) uses EOM branch selection with the single parent
cluster disabled. **Fine groups** uses terminal (leaf) branches, which may expose
smaller groups but leave more objects unassigned. **Allow one broad group** restores
the previous EOM behavior explicitly; it is disabled for leaf selection. Settings
are passed to the read-only worker and included in its cache key.
The representative is the medoid (smallest mean within-group distance); the
gallery shows its closest group members, using a shared physical scale within
that group. Mark the whole group, mark unassigned objects, or open a full-screen
plot colored by the discovered labels. HDBSCAN membership strength is not a
biological confidence estimate or an empirical resampling-stability score.

Primary/adjusted/all-method filters and the **Same objects across methods**
option control comparison coverage. These work with cell and organoid levels;
cell-only methods remain unavailable for organoids. Partial method coverage is
respected, excluded cells are omitted, and the original analysis groups and saved
dataset are unchanged. Ordinary sidebar filters refit discovery on the resulting
cohort. The query-only other-stack option does not alter density-group fitting.
Group labels in expanded plot windows are snapshots from the time they opened.
Colors repeat after the available palette, so use group numbers to identify groups.

The previous default allowed the parent cluster and often returned one broad
group. On the 744 included, QC-passing cells, minimum 10 / density neighbors 5,
the updated Balanced default returns six morphology groups (475 unassigned),
two Point-MAE adapted pose-average groups (658 unassigned), and four Lite3D
invariant groups (457 unassigned). Fine selection returns nine, two, and seven
groups respectively. The diagnostic, including dataset hash and other settings,
is in `results/atlas_interactive_checks/group_selection_diagnostic.json`.
The implementation does not force discrete biological types where the
descriptors do not show density separation. Nearest-shape galleries remain useful
when shape variation is continuous. No new biological validity is claimed.

## Execution and checks

Analyses run in a background Python process using the project's existing runtime.
Results are cached in `.cache/atlas-interactive`, keyed by dataset path, file
modification time/size, worker version and analysis settings/cohort. Membership
updates invalidate the cache by changing the dataset. The source dataset is
read-only for these analyses. There is no retraining or GPU requirement.
The gallery loads lazy rows and preserves native mesh/camera behavior.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p test_atlas_interactive.py -v
zsh native/OrganoidExplorer/run_checks.sh --shapes
zsh native/OrganoidExplorer/run_checks.sh
zsh native/OrganoidExplorer/build_app.sh
```

Six scientific tests cover PCA parity, degenerate dimensions, full-distance
clustering, representative selection, cosine distance, global CAJAL identities,
spatial fallback, exclusions and missing results. The native worker check covers
the Python bridge/cache, group-to-table marking and source preservation. A real
data check of projections and discovery for all 37 methods is recorded in
`results/atlas_interactive_checks/all_methods.json`.

HDBSCAN behavior follows the installed [scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).
