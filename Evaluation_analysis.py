# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Evaluation and submission analysis for DeepRacer
#
# This notebook has been built based on the `DeepRacer Log Analysis.ipynb` provided by the AWS DeepRacer Team. It has been reorganised and expanded to provide new views on the evaluation/racing data in a cleaner way, without the helper code which was moved into utility `.py` files.
#
# **You will find this notebook most useful for race submissions reviews and because of that it is mostly focusing on this goal.**
#
# ## Usage
#
# I am assuming here that you have already become familiar with `Training_analysis.ipynb`. Therefore descriptions that you will find here may be missing some bits if already described in there.
#
# Since this file can change in the future, I recommend that you make its copy and reorganize it to your liking. This way you will not lose your changes and you'll be able to add things as you please.
#
# **This notebook isn't complete.** What I find interesting in the logs may not be what you will find interesting and useful. I recommend you get familiar with the tools and try hacking around to get the insights that suit your needs.
#
# ## Contributions
#
# As usual, your ideas are very welcome and encouraged so if you have any suggestions either bring them to [the AWS DeepRacer Community](http://join.deepracing.io) or share as code contributions.
#
# ## Training environments
#
# Depending on whether you're running your evaluations through the console or using the local setup, and on which setup for local training you're using, your experience will vary. As much as I would like everything to be taylored to your configuration, there may be some problems that you may face. If so, please get in touch through [the AWS DeepRacer Community](http://join.deepracing.io).
#
# For race submissions it is much more straightforward.
#
# ## Requirements
#
# Before you start using the notebook, you will need to install some dependencies. If you haven't yet done so, have a look at [The README.md file](/edit/README.md#running-the-notebooks) to find what you need to install.
#
# Apart from the install, you also have to configure your programmatic access to AWS. Have a look at the guides below, AWS resources will lead you by the hand:
#
# AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html
#
# Boto Configuration: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
#
# ## Credits
#
# I would like to thank [the AWS DeepRacer Community](http://join.deepracing.io) for all the feedback about the notebooks. If you'd like, follow [my blog](https://codelikeamother.uk) where I tend to write about my experiences with AWS DeepRacer.
#
# # Log Analysis
#
# Let's get to it.
#
# ## Imports
#
# Run the imports block below:

# +

from deepracer.tracks import TrackIO, Track

from deepracer.logs import AnalysisUtils as au, \
    SimulationLogsIO as slio, \
    EvaluationUtils as eu, \
    PlottingUtils as pu

# Ignore deprecation warnings we have no power over
import warnings
warnings.filterwarnings('ignore')
# -

import pandas as pd

# ## Load waypoints for the track you want to run analysis on
#
# You will notice files for racing tracks. They are community best-effort versions made to make the visualisation in the logs less confusing. They may be slightly differing from reality, we don't know for sure. We do not have access to actual npy files that AWS use in the league.
#
# Tracks Available:

# +
# !ls tracks/

tu = TrackIO()

# +
track: Track = tu.load_track("reInvent2019_track")

track.road_poly
# -

# ## Load all race submission logs
#
# **WARNING:** If you do not specify `not_older_than` parameter, all evaluation logs will be downloaded. They aren't as big as the training logs, but there is a lot of them.
#
# That said you can download all and then it will only download new ones unless you use force=True.
#
# There are also `not_older_than` and `older_than` parameters so you can choose to fetch all logs from a given period and compare them against each other. Just remember memory is finite.
#
# As mentioned, this method always fetches a list of log streams and then downloads only ones that haven't been downloaded just yet. You can therefore use it to fetch that list and load all the files from the path provided.
#
# Side note: if you want to download evaluation logs from AWS DeepRacer Console, this will be a bit more tricky. Evaluation logs are grouped together with training logs in same group `/aws/robomaker/SimulationJobs` and there isn't an obvious way to recognise which ones they are. That said, in `Evaluation Run Analysis` section below you have the ability to download a single evaluation file.

# +
# For the purpose of generating the notebook in a reproducible way
# logs download has been commented out.
logs = [('/home/techis/deepracer-for-cloud/data/logs/robo3.log', 'sim-sample')]

# logs = cw.download_all_logs(
#     'logs/deepracer-eval-', 
#     '/aws/deepracer/leaderboard/SimulationJobs', 
#     not_older_than="2019-07-01 07:00", 
#     older_than="2019-07-01 12:00"
# )
# -

# Loads all the logs from the above time range
bulk = slio.load_a_list_of_logs(logs)

bulk=pd.DataFrame(bulk)

bulk



# ## Parse logs and visualize
#
# You will notice in here that reward graps are missing, as are many others from the training. These have been trimmed down for clarity.
#
# Do not get tricked though - this notebook provides features that the training one doesn't have, such as batch visualisation of race submission laps.
#
# Side note: Evaluation/race logs contain a reward field but it's not connected to your reward. It is there most likely to ensure logs have consistent structure to make their parsing easier. The value appears to be dependand on distance of the car from the centre of the track. As such it provides no value and is not visualised in this notebook.

# +
simulation_agg = au.simulation_agg(bulk, 'stream',is_eval=True)
complete_ones = simulation_agg[simulation_agg['progress']==100]

# This gives the warning about ptp method deprecation. The code looks as if np.ptp was used, I don't know how to fix it.
au.scatter_aggregates(simulation_agg, is_eval=True)
if complete_ones.shape[0] > 0:
    au.scatter_aggregates(complete_ones, "Complete ones", is_eval=True)

# +
#simulation_agg = au.simulation_agg(bulk, is_eval=True)
complete_ones = simulation_agg[simulation_agg['progress']==100]

# This gives the warning about ptp method deprecation. The code looks as if np.ptp was used, I don't know how to fix it.
#au.scatter_aggregates(simulation_agg, is_eval=True)
if complete_ones.shape[0] > 0:
    au.scatter_aggregates(complete_ones, "Complete ones", is_eval=True)
# -

# ## Data in tables

# View fifteen most progressed attempts
simulation_agg.nlargest(15, 'progress')

# View fifteen fastest complete laps
complete_ones.nsmallest(15, 'time')


