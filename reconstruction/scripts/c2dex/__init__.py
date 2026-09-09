"""C2Dex contact-stabilization port for the do-as-i-do reconstruction pipeline.

Implements the image-space stage of C2Dex Sec. III-A(b), "Cross-Frame Contact
Stabilization": rendering the hand and object silhouettes S_h,t / S_o,t under
the estimated camera, intersecting them into the candidate contact region
S_c,t, and retaining the hand vertices whose projections fall inside it.

The downstream stages of the paper (ray-cast object-side observations x_t,i,
canonical-space alignment, segment partition, DBSCAN, medoid) consume the
output of this stage and are not implemented here.
"""
