
import dash
import numpy as np
import socket
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from pycuda_plus.utils.monitor import start_memory_monitoring, memory_data, data_lock
from pycuda_plus.core.context import CudaContextManager


class GpuDashboard:
    # Modern color palette
    COLORS = {
        'background': '#f8fafc',
        'card': '#ffffff',
        'primary': '#3b82f6',
        'secondary': '#64748b',
        'accent': '#f43f5e',
        'success': '#22c55e',
        'text': '#1e293b',
        'lightText': '#64748b'
    }
    
    CHART_COLORS = ['#3b82f6', '#f43f5e', '#22c55e', '#eab308', '#a855f7']

    def __init__(self, kernel_comparator, rows, cols):
        """Initialize the GPU Dashboard."""
        self.kernel_comparator = kernel_comparator
        self.rows = rows
        self.cols = cols
        self.app = dash.Dash(
            __name__, 
            suppress_callback_exceptions=True,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
        )

        if not hasattr(GpuDashboard, 'memory_monitoring_started'):
            start_memory_monitoring()
            GpuDashboard.memory_monitoring_started = True

        self.layout = self.create_layout()
        self.app.layout = self.layout
        self.setup_callbacks()

    def create_card(self, children, width='100%'):
        """Create a styled card component."""
        return html.Div(
            children,
            style={
                'backgroundColor': self.COLORS['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 1px 3px 0 rgb(0 0 0 / 0.1)',
                'width': width,
                'margin': '10px'
            }
        )

    def create_layout(self):
        """Define the layout of the dashboard."""
        return html.Div(style={
            'backgroundColor': self.COLORS['background'],
            'minHeight': '100vh',
            'padding': '20px'
        }, children=[
            # Header
            html.Div(style={
                'textAlign': 'center',
                'marginBottom': '30px',
                'padding': '20px',
                'backgroundColor': self.COLORS['card'],
                'borderRadius': '8px',
                'boxShadow': '0 1px 3px 0 rgb(0 0 0 / 0.1)'
            }, children=[
                html.H1('GPU Utilization Dashboard', style={
                    'color': self.COLORS['text'],
                    'fontSize': '2.25rem',
                    'fontWeight': '600',
                    'margin': '0'
                }),
                html.P('Real-time monitoring and performance analysis', style={
                    'color': self.COLORS['lightText'],
                    'margin': '10px 0 0 0'
                })
            ]),

            # Memory Usage Section
            html.Div(style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'justifyContent': 'space-between',
                'marginBottom': '20px'
            }, children=[
                self.create_card([
                    html.H3('Memory Distribution', style={
                        'color': self.COLORS['text'],
                        'marginTop': '0'
                    }),
                    dcc.Graph(id='memory-usage-pie-chart')
                ], width='30%'),
                
                self.create_card([
                    html.H3('Memory Usage Over Time', style={
                        'color': self.COLORS['text'],
                        'marginTop': '0'
                    }),
                    dcc.Graph(id='real-time-memory-usage')
                ], width='55%')
            ]),

            # Kernel Execution Times Section
            self.create_card([
                html.H3('Kernel Performance Analysis', style={
                    'color': self.COLORS['text'],
                    'marginTop': '0'
                }),
                dcc.Graph(id='kernel-execution-times'),
                
                # Controls
                html.Div(style={
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'marginTop': '20px',
                    'gap': '20px'
                }, children=[
                    html.Div([
                        html.Label("Rows", style={
                            'color': self.COLORS['text'],
                            'marginRight': '10px',
                            'fontWeight': '500'
                        }),
                        dcc.Input(
                            id="rows-input",
                            type="number",
                            value=self.rows,
                            style={
                                'padding': '8px',
                                'borderRadius': '4px',
                                'border': f'1px solid {self.COLORS["secondary"]}',
                                'width': '100px'
                            }
                        )
                    ]),
                    html.Div([
                        html.Label("Columns", style={
                            'color': self.COLORS['text'],
                            'marginRight': '10px',
                            'fontWeight': '500'
                        }),
                        dcc.Input(
                            id="cols-input",
                            type="number",
                            value=self.cols,
                            style={
                                'padding': '8px',
                                'borderRadius': '4px',
                                'border': f'1px solid {self.COLORS["secondary"]}',
                                'width': '100px'
                            }
                        )
                    ])
                ])
            ]),

            dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
        ])

    def plot_kernel_execution_times(self, execution_times):
        """Plot the kernel execution times with improved styling."""
        if not execution_times:
            return {'data': [], 'layout': go.Layout(title='No Data Available')}

        traces = []
        for i, (operation, times) in enumerate(execution_times.items()):
            traces.append(
                go.Bar(
                    x=[f"Run {j+1}" for j in range(min(10, len(times)))],
                    y=times[:10],
                    name=operation,
                    marker=dict(color=self.CHART_COLORS[i % len(self.CHART_COLORS)])
                )
            )

        layout = go.Layout(
            title=None,  # Title is handled by the card
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.COLORS['text']),
            barmode='group',
            xaxis=dict(
                title='Run Number',
                gridcolor=self.COLORS['secondary'],
                gridwidth=0.1,
                showline=True,
                linecolor=self.COLORS['secondary']
            ),
            yaxis=dict(
                title='Execution Time (s)',
                gridcolor=self.COLORS['secondary'],
                gridwidth=0.1,
                showline=True,
                linecolor=self.COLORS['secondary']
            ),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            )
        )

        return {'data': traces, 'layout': layout}

    def plot_memory_usage(self, used_memory, free_memory):
        """Generate a modernized memory usage pie chart."""
        return {
            'data': [go.Pie(
                labels=['Used Memory', 'Free Memory'],
                values=[used_memory, free_memory],
                hole=0.7,
                marker=dict(colors=[self.COLORS['accent'], self.COLORS['primary']]),
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(color=self.COLORS['text'])
            )],
            'layout': go.Layout(
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(t=50, b=50, l=10, r=10),
                height=400
            )
        }

    def plot_real_time_memory_usage(self, timestamps, used_memory, free_memory):
        """Generate a modernized real-time memory usage chart."""
        return {
            'data': [
                go.Scatter(
                    x=timestamps,
                    y=used_memory,
                    mode='lines',
                    name='Used Memory',
                    line=dict(color=self.COLORS['accent'], width=2),
                    fill='tozeroy',
                    fillcolor=f'rgba{tuple(int(self.COLORS["accent"][1:][i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}'
                ),
                go.Scatter(
                    x=timestamps,
                    y=free_memory,
                    mode='lines',
                    name='Free Memory',
                    line=dict(color=self.COLORS['primary'], width=2),
                    fill='tozeroy',
                    fillcolor=f'rgba{tuple(int(self.COLORS["primary"][1:][i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}'
                )
            ],
            'layout': go.Layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.COLORS['text']),
                xaxis=dict(
                    title='Time (s)',
                    showgrid=True,
                    gridcolor=self.COLORS['secondary'],
                    gridwidth=0.1,
                    showline=True,
                    linecolor=self.COLORS['secondary']
                ),
                yaxis=dict(
                    title='Memory Usage (bytes)',
                    showgrid=True,
                    gridcolor=self.COLORS['secondary'],
                    gridwidth=0.1,
                    showline=True,
                    linecolor=self.COLORS['secondary']
                ),
                legend=dict(
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)'
                ),
                margin=dict(l=50, r=20, t=20, b=50)
            )
        }

    def setup_callbacks(self):
        """Setup callbacks for updating the graphs."""
        self.app.callback(
            Output('kernel-execution-times', 'figure'),
            [Input('interval-component', 'n_intervals'), Input('rows-input', 'value'), Input('cols-input', 'value')]
        )(self.update_kernel_execution_times)

        self.app.callback(
            Output('memory-usage-pie-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )(self.update_memory_usage)

        self.app.callback(
            Output('real-time-memory-usage', 'figure'),
            Input('interval-component', 'n_intervals')
        )(self.update_real_time_memory_usage)
    
    def update_kernel_execution_times(self, n_intervals, rows, cols):
        """Update the kernel execution times graph with multiple runs."""
        print(f"Running kernel comparison for Rows: {rows}, Cols: {cols}")

        if not rows or not cols or rows <= 0 or cols <= 0:
            return {'data': [], 'layout': go.Layout(title="Invalid Input Data")}

        try:
            execution_times = {}
            with CudaContextManager() as _:
                for _ in range(5):  # Run the kernels 5 times
                    run_times = self.kernel_comparator.compare_kernels(rows, cols)

                    # Accumulate execution times
                    for key, times in run_times.items():
                        if key not in execution_times:
                            execution_times[key] = []
                        execution_times[key].extend(times)  # Add multiple runs

            if not execution_times or not isinstance(execution_times, dict):
                return {'data': [], 'layout': go.Layout(title="No Execution Data Available")}

            return self.plot_kernel_execution_times(execution_times)

        except Exception as e:
            print(f"Error updating kernel execution times: {e}")
            return {'data': [], 'layout': go.Layout(title="Error Plotting Data")}

    def update_memory_usage(self, n_intervals):
        """Update the memory usage pie chart."""
        with data_lock:
            used_memory = memory_data.get('used_memory', [0])[-1]
            free_memory = memory_data.get('free_memory', [0])[-1]

        return self.plot_memory_usage(used_memory, free_memory)

    def update_real_time_memory_usage(self, n_intervals):
        """Update the real-time memory usage graph."""
        with data_lock:
            timestamps = memory_data.get('timestamps', [])[-100:]
            used_memory = memory_data.get('used_memory', [])[-100:]
            free_memory = memory_data.get('free_memory', [])[-100:]

        if not timestamps:
            return {}

        return self.plot_real_time_memory_usage(timestamps, used_memory, free_memory)

    def find_available_port(self, start_port=8000, end_port=9000):
        """Find an available port within the specified range."""
        for port in range(start_port, end_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        raise Exception(f"No available ports found in range {start_port}-{end_port}.")

    def run(self):
        """Run the Dash app on an available port."""
        try:
            available_port = self.find_available_port()
            print(f"Running the Dash app on port {available_port}...")
            self.app.run_server(debug=True, port=available_port, use_reloader=False)
        except Exception as e:
            print(f"Error: {e}")

def plot_execution_times(kernel_comparator, rows, cols):
    """Helper function to start the GPU dashboard."""
    if not hasattr(kernel_comparator, 'compare_kernels'):
        print("Error: kernel_comparator is missing 'compare_kernels' method.")
        return

    GpuDashboard(kernel_comparator, rows=rows, cols=cols).run()
