"""
napari widgets for cell registration workflows.
"""

from typing import Optional, Annotated
from pathlib import Path
from enum import Enum
import numpy as np
import napari
from napari.types import ImageData, LabelsData
from napari.utils.notifications import show_info

from .core import (
    CellposeConfig,
    CellposeSegmenter,
    compute_cell_features,
    CellFeaturesConfig,
    greedy_match_cells,
    MatchingConfig,
    estimate_rigid_transform_from_matches,
)
from .core.validation import validate_matches, compute_match_quality_stats



class CellposeModel(Enum):
    """Cellpose model options."""
    CPSAM = "cpsam"
    CYTO = "cyto"
    NUCLEI = "nuclei"
    CYTO2 = "cyto2"
    CYTO3 = "cyto3"


def segment_cells_widget(
    viewer: napari.Viewer,
    images: list[ImageData],
    model: CellposeModel = CellposeModel.CPSAM,
    gpu: bool = False,
    diameter: float = 10,
    flow_threshold: Annotated[float, {"min": -10.0, "max": 10.0, "step": 0.1}] = 0.0,
    cellprob_threshold: Annotated[float, {"min": -10.0, "max": 10.0, "step": 0.1}] = 0.0,
    min_size: int = 15,
    save_masks: bool = False,
    output_dir: str = "./segmentation_output",
):
    """
    Segment cells using Cellpose.
    
    Supports batch processing of multiple images.
    
    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance
    images : list[ImageData]
        Input images (DAPI channel) - can select multiple layers
    model : CellposeModel
        Cellpose model selection
    gpu : bool
        Whether to use GPU acceleration
    diameter : float
        Expected cell diameter in pixels (0 = auto-detect)
    flow_threshold : float
        Flow threshold (higher = fewer cells)
    cellprob_threshold : float
        Cell probability threshold
    min_size : int
        Minimum cell size in pixels
    save_masks : bool
        Save segmentation masks to disk
    output_dir : str
        Output directory for saved masks
    """
    from pathlib import Path
    from tifffile import imwrite
    
    if not images:
        show_info("Please select at least one image layer.")
        return
    
    show_info(f"=== Starting Cell Segmentation ===")
    show_info(f"Model: {model.value} | Total images: {len(images)}")
    
    # Configure Cellpose
    config = CellposeConfig(
        gpu=gpu,
        pretrained_model=model.value,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    
    # Create segmenter
    segmenter = CellposeSegmenter(config)
    
    # Get the original layer names from viewer
    layer_names = [layer.name for layer in viewer.layers if hasattr(layer, 'data')]
    
    # Prepare output directory if saving
    if save_masks:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        show_info(f"Masks will be saved to: {output_path.absolute()}")
    
    # Process each image
    for idx, image in enumerate(images):
        progress = f"[{idx+1}/{len(images)}]"
        
        # Find matching layer name - FIX: avoid numpy array comparison ambiguity
        layer_name = f"Image_{idx+1}"
        img_array = np.asarray(image)
        for name in layer_names:
            layer = viewer.layers[name]
            if hasattr(layer, 'data'):
                try:
                    # Use shape and hash comparison instead of array_equal in boolean context
                    layer_data = np.asarray(layer.data)
                    if layer_data.shape == img_array.shape and np.array_equal(layer_data, img_array):
                        layer_name = name
                        break
                except:
                    continue
        
        show_info(f"{progress} Processing {layer_name}...")
        
        # Run segmentation
        mask, flows, styles = segmenter.segment_array(img_array)
        
        # Count cells
        n_cells = len(np.unique(mask)) - 1
        
        # Create mask name based on original image name
        mask_name = f"{layer_name}_mask"
        
        # Add mask to viewer
        viewer.add_labels(mask, name=mask_name, opacity=0.5)
        
        show_info(f"{progress} {mask_name}: Found {n_cells} cells")
        
        # Save mask if requested
        if save_masks:
            mask_filename = output_path / f"{layer_name}_mask.tif"
            imwrite(str(mask_filename), mask.astype(np.uint16))
            show_info(f"{progress} ✓ Saved: {mask_filename.name}")
    
    show_info(f"=== Segmentation Complete! Processed {len(images)} image(s) ===")


def registration_workflow_widget(
    viewer: napari.Viewer,
    image_round1: ImageData,
    image_round2: ImageData,
    mask_round1: LabelsData,
    mask_round2: LabelsData,
    use_advanced_registration: bool = True,
    use_neighbor_refinement: bool = True,
    neighbor_k: int = 5,
    gpu: bool = False,
    use_two_stage_matching: bool = True,
    position_weight: float = 3.0,
    distance_threshold: float = 2.5,
    spatial_window_size: float = 100.0,
    top_k: int = 100,
    min_area: int = 0,
    max_area: int = 0,
    save_results: bool = False,
    output_dir: str = "./registration_output",
):
    """
    Complete cell registration workflow using pre-segmented masks.
    
    Performs:
    1. Feature extraction from masks
    2. Cell matching (two-stage: coarse morphology-based, then fine position-based)
    3. Registration (transform estimation with optional neighbor refinement)
    4. Visualization of results with image overlay
    
    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance
    image_round1 : ImageData
        First round original image (for overlay visualization)
    image_round2 : ImageData
        Second round original image (to be transformed)
    mask_round1 : LabelsData
        First round segmentation mask (reference)
    mask_round2 : LabelsData
        Second round segmentation mask (to be registered)
    use_advanced_registration : bool
        Use advanced registration (Similarity transform + RANSAC). If False, uses simple translation.
    use_neighbor_refinement : bool
        Apply neighbor-based refinement to preserve local structure (only if use_advanced_registration=True)
    neighbor_k : int
        Number of nearest neighbors to use in refinement (default: 5)
    gpu : bool
        Whether to use GPU for any GPU-accelerated operations (currently not used in matching/registration)
    use_two_stage_matching : bool
        Use two-stage matching: first morphology-only to estimate global offset, then position-weighted
    position_weight : float
        Weight for spatial position in final matching (default: 3.0 for high spatial weight)
    distance_threshold : float
        Maximum match distance threshold (standardized units) - rejects matches beyond this
    spatial_window_size : float
        Spatial search window radius in pixels (default: 100). Cells can only match within this distance.
        Increase if images have large displacement; decrease for higher precision.
    top_k : int
        Maximum number of matches to return (default: 100)
    min_area : int
        Minimum cell area for filtering (0 = no filter)
    max_area : int
        Maximum cell area for filtering (0 = no filter)
    save_results : bool
        Save registration results (CSV and masks) to disk
    output_dir : str
        Output directory for saved results
    """
    from .core.visualization import warp_mask_to_image2
    from skimage.transform import AffineTransform, warp, SimilarityTransform
    from skimage.measure import ransac
    from .core.matching import apply_transform_to_features, two_stage_match_cells
    from .core.point_registration import estimate_robust_transform, refine_transform_with_neighbors
    from pathlib import Path
    from tifffile import imwrite
    import pandas as pd
    
    # Import advanced registration if needed
    if use_advanced_registration:
        try:
            _ = refine_transform_with_neighbors
        except ImportError:
            show_info("⚠ Advanced registration not available, falling back to simple method")
            use_advanced_registration = False
    
    show_info("=== Starting Cell Registration Workflow ===")
    
    # Prepare output directory if saving
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        show_info(f"Results will be saved to: {output_path.absolute()}")
    
    # Convert to numpy arrays
    mask1 = np.asarray(mask_round1)
    mask2 = np.asarray(mask_round2)
    img1 = np.asarray(image_round1)
    img2 = np.asarray(image_round2)
    
    # Step 1: Feature extraction
    show_info("[1/5] Extracting features from Round 1...")
    feat_config = CellFeaturesConfig(
        min_area=min_area if min_area > 0 else None,
        max_area=max_area if max_area > 0 else None
    )
    feats1 = compute_cell_features(mask1, feat_config)
    n_cells1 = len(feats1)
    show_info(f"  ✓ Round 1: {n_cells1} cells after filtering")
    
    show_info("[2/5] Extracting features from Round 2...")
    feats2 = compute_cell_features(mask2, feat_config)
    n_cells2 = len(feats2)
    show_info(f"  ✓ Round 2: {n_cells2} cells after filtering")
    
    # Add all centroids to viewer
    if not feats1.empty:
        pts1 = feats1[["centroid_y", "centroid_x"]].to_numpy()
        viewer.add_points(pts1, name="All Points Round 1", size=4, face_color="cyan", opacity=0.5)
    
    if not feats2.empty:
        pts2 = feats2[["centroid_y", "centroid_x"]].to_numpy()
        viewer.add_points(pts2, name="All Points Round 2", size=4, face_color="magenta", opacity=0.5)
    
    # Step 2: Matching (Two-Stage Algorithm)
    show_info("[3/5] Matching cells...")
    
    if use_two_stage_matching and n_cells1 >= 20 and n_cells2 >= 20:
        show_info("  → Using two-stage matching to handle global offset...")
        
        # Stage 1: Coarse matching using morphology only (ignore position)
        show_info("    [Stage 1] Morphology-based coarse matching...")
        config_coarse = MatchingConfig(
            position_weight=0.0,  # Ignore position in coarse matching
            top_k=min(50, n_cells1, n_cells2),
            distance_threshold=2.0,  # Reasonable morphology distance
            spatial_window_size=spatial_window_size * 2.0,  # Use 2x window for coarse stage
        )
        coarse_matches = greedy_match_cells(feats1, feats2, config_coarse)
        show_info(f"    ✓ Coarse: {len(coarse_matches)} morphology-similar cells")
        
        # Use best coarse matches to estimate global offset
        if len(coarse_matches) >= 3:
            # Take top 20 best coarse matches for offset estimation
            n_for_offset = min(20, len(coarse_matches))
            best_coarse = coarse_matches.nsmallest(n_for_offset, 'distance')
            
            # Calculate average offset using MEDIAN (more robust than mean)
            pts1_coarse = feats1.loc[best_coarse['idx1'], ['centroid_x', 'centroid_y']].to_numpy()
            pts2_coarse = feats2.loc[best_coarse['idx2'], ['centroid_x', 'centroid_y']].to_numpy()
            offset = np.median(pts1_coarse - pts2_coarse, axis=0)  # Median is robust to outliers
            
            show_info(f"    ✓ Estimated global offset: ({offset[0]:.1f}, {offset[1]:.1f}) pixels")
            
            # Apply offset to Round 2 features
            feats2_aligned = feats2.copy()
            feats2_aligned['centroid_x'] = feats2['centroid_x'] + offset[0]
            feats2_aligned['centroid_y'] = feats2['centroid_y'] + offset[1]
            # Recalculate normalized positions
            h, w = mask1.shape
            feats2_aligned['pos_x_norm'] = feats2_aligned['centroid_x'] / float(max(w, 1))
            feats2_aligned['pos_y_norm'] = feats2_aligned['centroid_y'] / float(max(h, 1))
        else:
            show_info("    ⚠ Not enough coarse matches, skipping offset correction")
            feats2_aligned = feats2
        
        # Stage 2: Fine matching with HIGH position weight on aligned features
        show_info(f"    [Stage 2] Position-weighted fine matching (weight={position_weight}, window={spatial_window_size:.0f}px)...")
        config_fine = MatchingConfig(
            position_weight=position_weight,  # High position weight after alignment
            top_k=top_k,
            distance_threshold=distance_threshold,
            spatial_window_size=spatial_window_size,  # Apply spatial window constraint
        )
        matches = greedy_match_cells(feats1, feats2_aligned, config_fine)
        show_info(f"    ✓ Fine: {len(matches)} high-quality matches")
        
    else:
        # Single-stage matching (fallback or when not enough cells)
        if n_cells1 < 20 or n_cells2 < 20:
            show_info("  → Too few cells, using single-stage matching...")
        else:
            show_info("  → Single-stage matching (two-stage disabled)...")
        
        config = MatchingConfig(
            position_weight=position_weight,
            top_k=top_k,
            distance_threshold=distance_threshold,
            spatial_window_size=spatial_window_size,  # Apply spatial window constraint
        )
        matches = greedy_match_cells(feats1, feats2, config)
    
    feats2_for_validation = feats2
    if use_two_stage_matching:
        match_result = two_stage_match_cells(
            feats1,
            feats2,
            mask1.shape,
            position_weight=position_weight,
            top_k=top_k,
            distance_threshold=distance_threshold,
            spatial_window_size=spatial_window_size,
        )
        feats2_for_validation = match_result.aligned_df2
        matches = match_result.matches
        if not match_result.coarse_matches.empty:
            show_info(f"    Coarse: {len(match_result.coarse_matches)} morphology-similar cells")
            show_info(
                "    Estimated global offset: "
                f"({match_result.coarse_offset_xy[0]:.1f}, {match_result.coarse_offset_xy[1]:.1f}) pixels"
            )
        show_info(f"    Fine: {len(matches)} high-quality matches")
    
    # Validate matches
    config_for_validation = MatchingConfig(
        distance_threshold=distance_threshold,
        min_confidence=0.2,
        max_feature_diff=0.7,
    )
    matches = validate_matches(matches, feats1, feats2_for_validation, config_for_validation)
    
    # Compute and display match quality statistics
    stats = compute_match_quality_stats(matches)
    show_info(f"  ✓ Final matches: {stats['n_matches']}")
    if stats['n_matches'] > 0:
        show_info(f"    Distance: min={stats['distance_min']:.2f}, max={stats['distance_max']:.2f}, mean={stats['distance_mean']:.2f}")
    
    # Visualize matches as lines with color-coded quality
    if not matches.empty:
        # Create line segments connecting matched cells
        matched_pts1 = feats1.loc[matches["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
        matched_pts2 = feats2.loc[matches["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
        
        # Add matched feature points with unique colors
        viewer.add_points(matched_pts1, name="Feature Points Round 1", size=8, face_color="yellow")
        viewer.add_points(matched_pts2, name="Feature Points Round 2", size=8, face_color="orange")
        
        # Create lines with uniform color (no quality judgment)
        lines = []
        
        # Calculate pixel distances for statistics only
        pixel_distances = np.linalg.norm(matched_pts1 - matched_pts2, axis=1)
        
        # Create all lines
        for i in range(len(matches)):
            lines.append([matched_pts1[i], matched_pts2[i]])
        
        # Display statistics
        show_info(f"    Match line statistics:")
        show_info(f"      Distance: min={pixel_distances.min():.1f}px, max={pixel_distances.max():.1f}px, mean={pixel_distances.mean():.1f}px")
        
        # Add all lines in uniform cyan color
        viewer.add_shapes(
            lines,
            shape_type='line',
            edge_width=1,
            edge_color='cyan',
            name='Match Lines'
        )
    
    # Step 3: Registration
    show_info("[4/5] Estimating registration transform...")
    if len(matches) < 3:
        show_info(f"  ✗ Warning: Only {len(matches)} matches found. Need at least 3 for transform estimation.")
        return
    
    # Prepare matched point arrays (y,x format for image coords, swap to x,y for skimage)
    pts_r1_yx = feats1.loc[matches["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
    pts_r2_yx = feats2.loc[matches["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
    pts_r1_xy = pts_r1_yx[:, ::-1]  # swap to x,y
    pts_r2_xy = pts_r2_yx[:, ::-1]
    
    if use_advanced_registration:
        show_info("  → Using advanced registration (Similarity + RANSAC)...")
        # Estimate similarity transform with RANSAC
        model_robust, inliers = ransac(
            (pts_r2_xy, pts_r1_xy),
            SimilarityTransform,
            min_samples=3,
            residual_threshold=5.0,
            max_trials=100,
        )
        
        if inliers is None or inliers.sum() < 3:
            show_info("  ✗ RANSAC failed, falling back to simple method")
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
            affine_transform = AffineTransform(matrix=np.eye(3))
            affine_transform.params[:2, 2] = transform.translation[::-1]  # only translation
        else:
            show_info(f"  ✓ RANSAC: {inliers.sum()}/{len(inliers)} inliers")
            affine_transform = model_robust
            
            # Apply neighbor refinement if requested
            if use_neighbor_refinement:
                show_info(f"  → Refining with k={neighbor_k} neighbors...")
                affine_transform = refine_transform_with_neighbors(
                    pts_r1_xy,
                    pts_r2_xy,
                    initial=affine_transform,
                    k=neighbor_k,
                    neighbor_weight=1.0,
                    landmark_weight=10.0,
                    max_theta_deg=10.0,
                    max_translation=30.0,
                    max_scale_change=0.05,
                )
                show_info("  ✓ Refinement complete!")
    else:
        # Simple translation-only method
        show_info("  → Using simple translation-only registration...")
        transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
        affine_transform = AffineTransform(matrix=np.eye(3))
        affine_transform.params[:2, 2] = transform.translation[::-1]  # swap y,x to x,y
    
    if use_advanced_registration:
        robust = estimate_robust_transform(
            pts_r1_xy,
            pts_r2_xy,
            prefer_affine=True,
            residual_threshold=3.0,
            similarity_residual_threshold=5.0,
            max_trials=500,
        )
        affine_transform = robust.transform
        show_info(
            f"  Robust model: {robust.method} | "
            f"inliers={robust.inlier_count}/{len(matches)} | "
            f"median residual={robust.median_inlier_residual:.2f}px"
        )

        guided_feats2 = apply_transform_to_features(feats2, affine_transform, mask1.shape)
        guided_config = MatchingConfig(
            position_weight=max(position_weight, 4.0),
            top_k=top_k,
            distance_threshold=None if distance_threshold is None else distance_threshold + 0.5,
            spatial_window_size=None if spatial_window_size is None else min(spatial_window_size, 60.0),
        )
        guided_matches = greedy_match_cells(feats1, guided_feats2, guided_config)
        guided_matches = validate_matches(guided_matches, feats1, guided_feats2, config_for_validation)
        if len(guided_matches) >= 3:
            guided_r1_yx = feats1.loc[guided_matches["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
            guided_r2_yx = feats2.loc[guided_matches["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
            guided = estimate_robust_transform(
                guided_r1_yx[:, ::-1],
                guided_r2_yx[:, ::-1],
                prefer_affine=True,
                residual_threshold=3.0,
                similarity_residual_threshold=5.0,
                max_trials=500,
            )
            if guided.score() > robust.score():
                matches = guided_matches
                pts_r1_yx = guided_r1_yx
                pts_r2_yx = guided_r2_yx
                pts_r1_xy = guided_r1_yx[:, ::-1]
                pts_r2_xy = guided_r2_yx[:, ::-1]
                robust = guided
                affine_transform = robust.transform
                show_info(
                    f"  Guided rematch improved support: "
                    f"{robust.inlier_count}/{len(matches)} inliers"
                )

        if use_neighbor_refinement and robust.inlier_count >= 3:
            pts_fit_r1 = pts_r1_xy[robust.inliers]
            pts_fit_r2 = pts_r2_xy[robust.inliers]
            affine_transform = refine_transform_with_neighbors(
                pts_fit_r1,
                pts_fit_r2,
                initial=affine_transform,
                k=neighbor_k,
                neighbor_weight=1.0,
                landmark_weight=10.0,
                max_theta_deg=10.0,
                max_translation=30.0,
                max_scale_change=0.05,
                optimize_translation_only=(robust.method != "similarity"),
            )
            show_info("  Refinement complete!")
    
    show_info(f"  ✓ Registration complete!")
    
    # Step 4: Apply transformation and visualize
    show_info("[5/5] Applying transformation and creating overlay...")
    
    # Warp the image
    if img2.ndim == 2:
        # 2D grayscale image
        img2_warped = warp(
            img2.astype(float),
            inverse_map=affine_transform.inverse,
            output_shape=img1.shape,
            preserve_range=True
        ).astype(img2.dtype)
    elif img2.ndim == 3:
        # 3D image with channels - warp each channel separately
        img2_warped = np.zeros_like(img1)
        for c in range(min(img2.shape[-1], img1.shape[-1] if img1.ndim == 3 else 1)):
            if img1.ndim == 3:
                img2_warped[..., c] = warp(
                    img2[..., c].astype(float),
                    inverse_map=affine_transform.inverse,
                    output_shape=img1.shape[:2],
                    preserve_range=True
                ).astype(img2.dtype)
            else:
                img2_warped = warp(
                    img2[..., c].astype(float),
                    inverse_map=affine_transform.inverse,
                    output_shape=img1.shape,
                    preserve_range=True
                ).astype(img2.dtype)
                break
    
    # Add warped image to viewer
    viewer.add_image(img2_warped, name="Registered Map", opacity=0.5, colormap="green", blending="additive")
    
    # Also add transformed centroids for verification
    pts_r2_reg = affine_transform(pts_r2_xy)[:, ::-1]  # convert back to y,x
    viewer.add_points(pts_r2_reg, name="Registered Point", size=6, face_color="red", opacity=0.8)
    
    show_info("  ✓ Visualization complete!")
    
    # Save results if requested
    if save_results:
        show_info("Saving results...")
        
        # Save registered image
        img_filename = output_path / "registered_image.tif"
        imwrite(str(img_filename), img2_warped)
        show_info(f"  ✓ Saved: {img_filename.name}")
        
        # Save feature CSVs
        feats1_csv = output_path / "features_round1.csv"
        feats1.to_csv(feats1_csv, index=False)
        show_info(f"  ✓ Saved: {feats1_csv.name}")
        
        feats2_csv = output_path / "features_round2.csv"
        feats2.to_csv(feats2_csv, index=False)
        show_info(f"  ✓ Saved: {feats2_csv.name}")
        
        # Save matches
        matches_csv = output_path / "matches.csv"
        matches.to_csv(matches_csv, index=False)
        show_info(f"  ✓ Saved: {matches_csv.name} ({len(matches)} matches)")
        
        # Save registered centroids
        pts_r2_reg_df = pd.DataFrame({
            'centroid_y': pts_r2_reg[:, 0],
            'centroid_x': pts_r2_reg[:, 1]
        })
        reg_pts_csv = output_path / "registered_centroids_round2.csv"
        pts_r2_reg_df.to_csv(reg_pts_csv, index=False)
        show_info(f"  ✓ Saved: {reg_pts_csv.name}")
        
        # Save transform matrix
        transform_file = output_path / "transform_matrix.txt"
        with open(transform_file, 'w') as f:
            f.write("Affine Transform Matrix:\n")
            f.write(str(affine_transform.params))
        show_info(f"  ✓ Saved: {transform_file.name}")
    
    show_info("=== Registration Complete! ===")
    show_info(f"Check 'Registered Map' and 'Registered Point' layers for results.")
