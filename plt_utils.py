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

# +
class PlottingUtils:
    """Utilities to visualise track and the episodes
    """
    @staticmethod
    def print_border(ax, track: Track, color='lightgrey'):
        """Print track borders on the chart
        Arguments:
        ax - axes to plot borders on
        track - the track info to plot
        color - what color to plot the border in, default: lightgrey
        """
        line = LineString(track.center_line)
        PlottingUtils._plot_coords(ax, line)
        PlottingUtils._plot_line(ax, line, color)

        line = LineString(track.inner_border)
        PlottingUtils._plot_coords(ax, line)
        PlottingUtils._plot_line(ax, line, color)

        line = LineString(track.outer_border)
        PlottingUtils._plot_coords(ax, line)
        PlottingUtils._plot_line(ax, line, color)

    @staticmethod
    def plot_selected_laps(sorted_idx, df, track: Track, section_to_plot="episode"):
        """Plot n laps in the training, referenced by episode ids
        Arguments:
        sorted_idx - a datagram with ids to be plotted or a list of ids
        df - a datagram with all data
        track - track info for plotting
        secton_to_plot - what section of data to plot - episode/iteration
        """

        ids = sorted_idx

        if type(sorted_idx) is not list:
            ids = sorted_idx[section_to_plot].unique().tolist()

        n_laps = len(ids)

        fig = plt.figure(n_laps, figsize=(12, n_laps * 10))
        for i in range(n_laps):
            idx = ids[i]

            data_to_plot = df[df[section_to_plot] == idx]

            ax = fig.add_subplot(n_laps, 1, i + 1)

            ax.axis('equal')

            PlottingUtils.print_border(ax, track, color='cyan')

            data_to_plot.plot.scatter('x', 'y', ax=ax, s=10, c='blue')

        plt.show()
        plt.clf()

        # return fig

    @staticmethod
    def plot_evaluations(evaluations, track: Track, graphed_value='speed'):
        """Plot graphs for evaluations
        """
        from math import ceil

        streams = evaluations.sort_values(
            'tstamp', ascending=False).groupby('stream', sort=False)

        for _, stream in streams:
            episodes = stream.groupby('episode')
            ep_count = len(episodes)

            rows = ceil(ep_count / 3)
            columns = min(ep_count, 3)

            fig, axes = plt.subplots(rows, columns, figsize=(7*columns, 5*rows))
            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=7.0)

            for id, episode in episodes:
                if rows == 1:
                    ax = axes[id % 3]
                elif columns == 1:
                    ax = axes[int(id/3)]
                else:
                    ax = axes[int(id / 3), id % 3]

                PlottingUtils.plot_grid_world(
                    episode, track, graphed_value, ax=ax)

            plt.show()
            plt.clf()

    @staticmethod
    def plot_grid_world(
        episode_df,
        track: Track,
        graphed_value='speed',
        min_progress=None,
        ax=None
    ):
        """Plot a scaled version of lap, along with speed taken a each position
        """

        episode_df.loc[:, 'distance_diff'] = ((episode_df['x'].shift(1) - episode_df['x']) ** 2 + (
            episode_df['y'].shift(1) - episode_df['y']) ** 2) ** 0.5

        distance = np.nansum(episode_df['distance_diff'])
        lap_time = np.ptp(episode_df['tstamp'].astype(float))
        velocity = distance / lap_time
        average_speed = np.nanmean(episode_df['speed'])
        progress = np.nanmax(episode_df['progress'])

        if not min_progress or progress > min_progress:

            distance_lap_time = 'Distance, progress, lap time = %.2f m, %.2f %%, %.2f s' % (
                distance, progress, lap_time
            )
            speed_velocity = 'Average speed, velocity = %.2f (Gazebo), %.2f m/s' % (
                average_speed, velocity
            )

            fig = None
            if ax is None:
                fig = plt.figure(figsize=(16, 10))
                ax = fig.add_subplot(1, 1, 1)

            ax.set_facecolor('midnightblue')

            line = LineString(track.inner_border)
            PlottingUtils._plot_coords(ax, line)
            PlottingUtils._plot_line(ax, line)

            line = LineString(track.outer_border)
            PlottingUtils._plot_coords(ax, line)
            PlottingUtils._plot_line(ax, line)

            episode_df.plot.scatter('x', 'y', ax=ax, s=3, c=graphed_value,
                                    cmap=plt.get_cmap('plasma'))

            subtitle = '%s%s\n%s\n%s' % (
                ('Stream: %s, ' % episode_df['stream'].iloc[0]
                 ) if 'stream' in episode_df.columns else '',
                datetime.fromtimestamp(episode_df['tstamp'].iloc[0]),
                distance_lap_time,
                speed_velocity)
            ax.set_title(subtitle)

            if fig:
                plt.show()
                plt.clf()

    @staticmethod
    def plot_track(df, track: Track, value_field="reward", margin=1, cmap="hot"):
        """Plot track with dots presenting the rewards for steps
        """
        if df.empty:
            print("The dataframe is empty, check if you have selected an existing subset")
            return

        track_size = (np.asarray(track.size()) + 2*margin).astype(int) * 100
        track_img = np.zeros(track_size).transpose()

        x_coord = 0
        y_coord = 1

        # compensation moves car's coordinates in logs to start at 0 in each dimention
        x_compensation = track.outer_border[:, 0].min()
        y_compensation = track.outer_border[:, 1].min()

        for _, row in df.iterrows():
            x = int((row["x"] - x_compensation + margin) * 100)
            y = int((row["y"] - y_compensation + margin) * 100)

            # clip values that are off track
            if y >= track_size[y_coord]:
                y = track_size[y_coord] - 1

            if x >= track_size[x_coord]:
                x = track_size[x_coord] - 1

            track_img[y, x] = row[value_field]

        fig = plt.figure(1, figsize=(12, 16))
        ax = fig.add_subplot(111)

        shifted_track = Track("shifted_track", (track.waypoints -
                                                [x_compensation, y_compensation]*3 + margin) * 100)

        PlottingUtils.print_border(ax, shifted_track)

        plt.title("Reward distribution for all actions ")
        plt.imshow(track_img, cmap=cmap, interpolation='bilinear', origin="lower")

        plt.show()
        plt.clf()

    @staticmethod
    def plot_trackpoints(track: Track, annotate_every_nth=1):
        _, ax = plt.subplots(figsize=(20, 10))
        PlottingUtils.plot_points(ax, track.center_line, annotate_every_nth)
        PlottingUtils.plot_points(ax, track.inner_border, annotate_every_nth)
        PlottingUtils.plot_points(ax, track.outer_border, annotate_every_nth)
        ax.axis('equal')

        return ax

    @staticmethod
    def plot_points(ax, points, annotate_every_nth=1):
        ax.scatter(points[:-1, 0], points[:-1, 1], s=1)
        for i, p in enumerate(points):
            if i % annotate_every_nth == 0:
                ax.annotate(i, (p[0], p[1]))

    @staticmethod
    def _plot_coords(ax, ob):
        x, y = ob.xy
        ax.plot(x, y, '.', color='#999999', zorder=1)

    @staticmethod
    def _plot_bounds(ax, ob):
        x, y = zip(*list((p.x, p.y) for p in ob.boundary))
        ax.plot(x, y, '.', color='#000000', zorder=1)

    @staticmethod
    def _plot_line(ax, ob, color='cyan'):
        x, y = ob.xy
        ax.plot(x, y, color=color, alpha=0.7, linewidth=3, solid_capstyle='round',
                zorder=2)


class EvaluationUtils:
    @staticmethod
    def analyse_single_evaluation(eval_df, track: Track,
                                  min_progress=None):
        """Plot all episodes of a single evaluation
        """
        episodes = eval_df.groupby('episode').groups
        for e in episodes:
            PlottingUtils.plot_grid_world(
                eval_df[eval_df['episode'] == e], track, min_progress=min_progress)

    @staticmethod
    def analyse_multiple_race_evaluations(logs, track: Track, min_progress=None):
        for log in logs:
            EvaluationUtils.analyse_single_evaluation(
                SimulationLogsIO.load_pandas(log[0]), track, min_progress=min_progress)
# -


