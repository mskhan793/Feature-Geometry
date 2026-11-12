Status (dataset x encoder):

encoder                                                                   dinov3_l                                                             dinov2_l                                                             bioclip2                      siglip2
dataset                                                                                                                                                                                                                                                  
finbenthic2                                    cached (Feature-Geometry, verified)                                                                ready                                                                ready           missing checkpoint
aquamonitor                                    cached (Feature-Geometry, verified)                                                                ready                                                                ready           missing checkpoint
bioscan5m      cached in v1 (same checkpoint/transform; equivalence check pending)  cached in v1 (same checkpoint/transform; equivalence check pending)  cached in v1 (same checkpoint/transform; equivalence check pending)           missing checkpoint
df24                                                                missing images                                                       missing images                                                       missing images  missing images + checkpoint
fgvc_aircraft                                                                ready                                                                ready                                                                ready           missing checkpoint
food101                                                                      ready                                                                ready                                                                ready           missing checkpoint
cifar100                                                                     ready                                                                ready                                                                ready           missing checkpoint
oxford_pets                                                                  ready                                                                ready                                                                ready           missing checkpoint

Totals per encoder:

          images_to_encode  est_node_hours  est_gpu_hours  storage_gb_float32
encoder                                                                      
bioclip2           1840063           0.748          2.995               5.653
dinov2_l           1840063           0.748          2.995               7.537
dinov3_l            817620           0.332          1.331               7.537
siglip2            1840063           1.049          4.194               8.478
