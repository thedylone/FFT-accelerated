import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def merge_speedup_frame(
    frame, serial_frame, key="n", time_col="time_ms", frame_suffix="series"
):
    merged = frame.merge(
        serial_frame,
        on=key,
        suffixes=(f"_{frame_suffix}", "_serial"),
    )
    merged["speedup"] = (
        merged[f"{time_col}_serial"] / merged[f"{time_col}_{frame_suffix}"]
    )
    return merged


def plot_grouped_lines(
    ax,
    frames,
    x_col,
    y_col,
    *,
    baseline_y=None,
    baseline_label=None,
    xscale="log",
    xbase=2,
    yscale=None,
    xlabel="",
    ylabel="",
    title="",
    legend_kwargs=None,
    savepath=None,
):
    if baseline_y is not None:
        ax.axhline(
            y=baseline_y,
            linestyle="--",
            color="gray",
            label=baseline_label,
        )

    for frame, color, marker, label, markersize in frames:
        sns.lineplot(
            ax=ax,
            x=frame[x_col],
            y=frame[y_col],
            marker=marker,
            markersize=markersize,
            color=color,
            label=label,
        )

    if xscale:
        ax.set_xscale(xscale, base=xbase)
    if yscale:
        ax.set_yscale(yscale)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(**(legend_kwargs or {}))

    if savepath:
        plt.tight_layout()
        plt.savefig(savepath)
    plt.show()


def compute_multiscale_speedup_series(
    df, group_col="threads", n_col="n", time_col="time_ms", cmap=None
):
    groups_all = np.array(sorted(df[group_col].unique()))
    max_group = int(groups_all.max())
    n_values = sorted(df[n_col].unique())
    norm = plt.Normalize(vmin=np.log2(n_values[0]), vmax=np.log2(n_values[-1]))
    cmap = cmap or plt.cm.viridis
    series = []
    for n_val in n_values:
        subset = df[df[n_col] == n_val]
        baseline_rows = subset[subset[group_col] == 1]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows[time_col].values[0]
        groups = np.array(sorted(subset[group_col].unique()))
        speedups = []
        for g in groups:
            rows = subset[subset[group_col] == g]
            if rows.empty:
                continue
            speedups.append(baseline / rows[time_col].values[0])
        if len(groups) < 2:
            continue
        color = cmap(norm(np.log2(n_val)))
        series.append((groups, np.array(speedups), color))
    return series, groups_all, max_group, norm, cmap


def compute_multiscale_efficiency_series(
    df, group_col="threads", n_col="n", time_col="time_ms", cmap=None
):
    groups_all = np.array(sorted(df[group_col].unique()))
    max_group = int(groups_all.max())
    n_values = sorted(df[n_col].unique())
    norm = plt.Normalize(vmin=np.log2(n_values[0]), vmax=np.log2(n_values[-1]))
    cmap = cmap or plt.cm.viridis
    series = []
    for n_val in n_values:
        n_per_group = int(n_val / max_group)
        baseline_rows = df[(df[group_col] == 1) & (df[n_col] == n_per_group)]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows[time_col].values[0]
        valid_groups = []
        eff = []
        for g in groups_all:
            rows = df[
                (df[group_col] == g) & (df[n_col] == int(g) * n_per_group)
            ]
            if rows.empty:
                continue
            valid_groups.append(g)
            eff.append(baseline / rows[time_col].values[0])
        if len(valid_groups) < 2:
            continue
        color = cmap(norm(np.log2(n_val)))
        series.append((np.array(valid_groups), np.array(eff), color))
    return series, groups_all, max_group, norm, cmap


def draw_multiscale_series(
    ax,
    series,
    groups_all,
    cmap,
    norm,
    fill_y0=0,
    xlabel="x",
    ylabel="y",
    title="",
    cbar_label=None,
    vline=None,
    savepath=None,
    y_log=False,
    x_log=True,
):
    if vline is not None:
        ax.axhline(y=vline, linestyle="--", color="gray", label="Ideal")

    for step in range(1, len(groups_all)):
        scored_series = []
        for xvals, yvals, color in series:
            if len(yvals) > step:
                scored_series.append((abs(yvals[step]), xvals, yvals, color))

        for _, xvals, yvals, color in sorted(scored_series, reverse=True):
            ax.fill_between(
                xvals[step - 1 : step + 1],
                fill_y0,
                yvals[step - 1 : step + 1],
                color=color,
                alpha=0.25,
            )

    for xvals, yvals, color in series:
        sns.lineplot(ax=ax, x=xvals, y=yvals, marker="o", color=color, lw=2)

    if x_log:
        ax.set_xscale("log", base=2)
    if y_log:
        ax.set_yscale("log", base=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if cbar_label:
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(cbar_label)

    if savepath:
        plt.tight_layout()
        plt.savefig(savepath)
    plt.show()


def plot_speedup_vs_serial(
    ax,
    frame,
    serial_frame,
    *,
    key="n",
    time_col="time_ms",
    color,
    marker,
    label,
    markersize=3,
    xscale="log",
    xbase=2,
    xlabel="n",
    ylabel="Speedup over Serial",
    title="",
    legend_kwargs=None,
    savepath=None,
):
    merged = merge_speedup_frame(
        frame, serial_frame, key=key, time_col=time_col, frame_suffix="impl"
    )
    sns.lineplot(
        ax=ax,
        x=merged[key],
        y=merged["speedup"],
        marker=marker,
        markersize=markersize,
        color=color,
        label=label,
    )
    if xscale:
        ax.set_xscale(xscale, base=xbase)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(**(legend_kwargs or {}))
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath)
    return merged


def compute_grouped_speedup_series(
    frame,
    serial_frame,
    *,
    group_col,
    key="n",
    time_col="time_ms",
    frame_suffix="impl",
):
    merged = merge_speedup_frame(
        frame,
        serial_frame,
        key=key,
        time_col=time_col,
        frame_suffix=frame_suffix,
    )
    group_values = sorted(frame[group_col].unique())
    series = []
    for group_value in group_values:
        subset = merged[merged[group_col] == group_value]
        if subset.empty:
            continue
        series.append((group_value, subset))
    return merged, series


def plot_grouped_speedup_vs_serial(
    ax,
    frame,
    serial_frame,
    *,
    group_col,
    colors,
    key="n",
    time_col="time_ms",
    marker="o",
    label_template="{value}",
    baseline_label="Serial baseline",
    xlabel="n",
    ylabel="Speedup over Serial",
    title="",
    legend_kwargs=None,
    savepath=None,
):
    merged, grouped_series = compute_grouped_speedup_series(
        frame,
        serial_frame,
        group_col=group_col,
        key=key,
        time_col=time_col,
        frame_suffix="impl",
    )

    group_values = sorted(frame[group_col].unique())
    color_map = {}
    for index, group_value in enumerate(group_values):
        color_map[group_value] = colors[index]

    # Build series tuples: (xvals, yvals, color, label)
    series = []
    for group_value, subset in grouped_series:
        xvals = subset[key].values
        yvals = subset["speedup"].values
        color = color_map[group_value]
        label = label_template.format(value=group_value)
        series.append((xvals, yvals, color, label))

    # Get all unique x values and sort them
    x_all = np.array(sorted(merged[key].unique()))

    # Fill between pattern: step through x_all (exactly like multiscale)
    for step in range(1, len(x_all)):
        scored_series = []
        for xvals, yvals, color, label in series:
            if len(yvals) > step:
                scored_series.append(
                    (abs(yvals[step]), xvals, yvals, color, label)
                )

        for _, xvals, yvals, color, label in sorted(
            scored_series, reverse=True
        ):
            ax.fill_between(
                xvals[step - 1 : step + 1],
                1,
                yvals[step - 1 : step + 1],
                color=color,
                alpha=0.5,
            )

    # Draw lines
    for xvals, yvals, color, label in series:
        sns.lineplot(
            ax=ax,
            x=xvals,
            y=yvals,
            marker=marker,
            markersize=3,
            color=color,
            label=label,
        )

    ax.axhline(y=1, linestyle="--", color="gray", label=baseline_label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(**(legend_kwargs or {}))
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath)
    plt.show()
    return merged
