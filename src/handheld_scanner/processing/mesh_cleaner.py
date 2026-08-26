import os
import time
import numpy as np
import open3d as o3d


class MeshProcessor:
    @staticmethod
    def clean_mesh(mesh, min_cluster_size=200, smooth_iterations=6):
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_non_manifold_edges()

        # Remove small detached artifacts
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        triangles_to_remove = np.zeros(len(mesh.triangles), dtype=bool)
        for c_idx, count in enumerate(cluster_n_triangles):
            if count < min_cluster_size:
                triangles_to_remove |= (triangle_clusters == c_idx)

        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

        if smooth_iterations > 0:
            mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations, lambda_filter=0.5, mu=-0.53)

        mesh.compute_vertex_normals()
        return mesh

    @staticmethod
    def export(mesh, output_dir, prefix="scan"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        obj_path = os.path.join(output_dir, f"{prefix}_{timestamp}.obj")
        stl_path = os.path.join(output_dir, f"{prefix}_{timestamp}.stl")

        o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_colors=True)
        o3d.io.write_triangle_mesh(stl_path, mesh)
        return obj_path, stl_path
