from tapnet import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from tqdm import tqdm
import functools
from ert import ert_detect
from pipnet import pipnet_detect
from tapir_utils import preprocess_frames, construct_initial_causal_state, build_online_model_init, build_online_model_predict, postprocess_occlusions

project_dir = "/home/zuoxy/facetrack/"
vid_id = "005"
data_dir = project_dir + "data/"
# vid_path = data_dir + "300vw/300VW_Dataset_2015_12_14/"
vid_path = data_dir + "custom/bean.mp4"

def tracker(project_dir, vid_path, detector):

  #Load Checkpoint {form-width: "25%"}
  tapir_dir = "/home/zuoxy/tapnet/"
  checkpoint_path = tapir_dir + 'checkpoints/causal_tapir_checkpoint.npy'
  ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
  params, state = ckpt_state['params'], ckpt_state['state']

  #load video
  out_dir = project_dir + "assets/videos/"
  video = media.read_video(vid_path)
  height, width = video.shape[1:3]

  #detect facial landmark points on the first frame
  if(detector == "ert"):
    points = ert_detect(project_dir, vid_path)
  elif(detector == "pipnet"):
    points = pipnet_detect(project_dir, vid_path)

  # Internally, the tapir model has three stages of processing: computing
  # image features (get_feature_grids), extracting features for each query point
  # (get_query_features), and estimating trajectories given query features and
  # the feature grids where we want to track (estimate_trajectories).  
  online_init = hk.transform_with_state(build_online_model_init)
  online_init_apply = jax.jit(online_init.apply)

  online_predict = hk.transform_with_state(build_online_model_predict)
  online_predict_apply = jax.jit(online_predict.apply)

  rng = jax.random.PRNGKey(42)
  online_init_apply = functools.partial(
      online_init_apply, params=params, state=state, rng=rng
  )
  online_predict_apply = functools.partial(
      online_predict_apply, params=params, state=state, rng=rng
  )

  # Progressively Predict Sparse Point Tracks 
  resize_height = 256  
  resize_width = 256  
  num_points = 68  

  frames = media.resize_video(video, (resize_height, resize_width))
  # query_points = sample_random_points(0, frames.shape[1], frames.shape[2], num_points)
  query_points = points
  query_features, _ = online_init_apply(frames=preprocess_frames(frames[None, None, 0]), query_points=query_points[None])
  causal_state = construct_initial_causal_state(query_points.shape[0], len(query_features.resolutions) - 1)

  # Predict point tracks frame by frame
  predictions = []
  for i in range(frames.shape[0]):
    # print(i)
    (prediction, causal_state), _ = online_predict_apply(
        frames=preprocess_frames(frames[None, None, i]),
        query_features=query_features,
        causal_context=causal_state,
    )
    predictions.append(prediction)

  tracks = np.concatenate([x['tracks'][0] for x in predictions], axis=1)
  occlusions = np.concatenate([x['occlusion'][0] for x in predictions], axis=1)
  expected_dist = np.concatenate([x['expected_dist'][0] for x in predictions], axis=1)

  visibles = postprocess_occlusions(occlusions, expected_dist)

  # Visualize sparse point tracks
  tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
  video_viz = viz_utils.paint_point_track(video, tracks, visibles)
  media.write_video(out_dir+'output.mp4', video_viz, fps=20) 

tracker(project_dir, vid_path, detector="pipnet")