// charts.js - Chart rendering using Plotly or Chart.js
function renderChart(containerId, data, layout) {
    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot(containerId, data, layout);
    } else {
        console.warn('Plotly not loaded.');
    }
}
