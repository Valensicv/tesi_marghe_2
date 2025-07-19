import plotly.graph_objects as go

def create_waterfall_figure(
    name="20",
    orientation="v",
    measure=None,
    x=None,
    textposition="outside",
    text=None,
    y=None,
    connector=None,
    **kwargs
):
    """
    Create a Plotly Waterfall Figure with customizable arguments.

    Parameters
    ----------
    name : str
        Name of the waterfall trace.
    orientation : str
        Orientation of the waterfall ("v" or "h").
    measure : list
        List of measure types for each bar.
    x : list
        List of x-axis labels.
    textposition : str
        Position of the text labels.
    text : list
        List of text labels for each bar.
    y : list
        List of y values for each bar.
    connector : dict
        Connector line style dictionary.
    **kwargs : dict
        Additional keyword arguments for go.Waterfall.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The resulting Waterfall figure.
    """
    if measure is None:
        measure = ["relative", "relative", "total", "relative", "relative", "total"]
    if x is None:
        x = ["Sales", "Consulting", "Net revenue", "Purchases", "Other expenses", "Profit before tax"]
    if text is None:
        text = ["+60", "+80", "", "-40", "-20", "Total"]
    if y is None:
        y = [60, 80, 0, -40, -20, 0]
    if connector is None:
        connector = {"line": {"color": "rgb(63, 63, 63)"}}

    fig = go.Figure(go.Waterfall(
        name=name,
        orientation=orientation,
        measure=measure,
        x=x,
        textposition=textposition,
        text=text,
        y=y,
        connector=connector,
        **kwargs
    ))
    return fig