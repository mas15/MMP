// Global parameters:
// do not resize the chart canvas when its container does (keep at 600x400px)
Chart.defaults.global.responsive = false;
// define the chart data
window.onload = function() {

{% set labels, values, tweets = graph_data %}

var tweets_per_date = {
    {% for date, tweets in tweets.items() %}
        "{{date}}": {{tweets|tojson|safe}},
    {% endfor %}
}

// get chart canvas
var canvas = document.getElementById("currencyChart");

canvas.style.width='100%';
canvas.width  = canvas.offsetWidth;

var ctx = canvas.getContext("2d");

var gradient = ctx.createLinearGradient(0, 0, 0, 400);
gradient.addColorStop(0, 'rgba(204, 255, 153,1)');
gradient.addColorStop(1, 'rgba(51, 102, 0,0.6)');

var chartData = {
  labels : [{% for d in labels %}
            "{{d}}",
            {% endfor %}],
  datasets : [{
      fill: true,
      lineTension: 0.1,
      backgroundColor: gradient,
      borderColor: "rgba(0, 128, 0,1)",
      borderCapStyle: 'butt',
      borderJoinStyle: 'miter',
      pointBorderColor: "rgba(0, 128, 0,1)",
      pointBackgroundColor: "#fff",
      pointBorderWidth: 1,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "rgba(0, 128, 0,1)",
      pointHoverBorderColor: "rgba(220,220,220,1)",
      pointHoverBorderWidth: 2,
      pointRadius: 1,
      pointHitRadius: 10,
      data : [{% for item in values %}
              {{item}},
              {% endfor %}],
      spanGaps: false,

  }]
}

// create the chart using the chart canvas
var currencyChart = new Chart(ctx, {
  type: 'line',
  data: chartData,
  options: {
         legend: {
            display: false
         },
         scales: {
            xAxes: [{
                type: 'time',
                time: {
                    unit: 'month',
                },
            }],
        },
        tooltips: {
          enabled: true,
          mode: 'single',
          callbacks: {
            label: function(tooltipItems, data) {
                     var price = tooltipItems.yLabel;
                     var date = tooltipItems.xLabel;
                     return tweets_per_date[date];
                    // return  price + ' degrees';
                   }
          }
        },
      }
});




{% for change in ["Down", "NC", "Up"] %}
    {% set features_with_values = features_data[change] %}
    {% set features, values = features_with_values|unzip %}

    var canvas = document.getElementById("featuresChart{{ change }}");
    var ctx = canvas.getContext("2d");

    var chartColors = []
    var r = 70;
    var g = 128;
    var b = 0;
    for(var i=0; i<20;i++){
        chartColors[i] = 'rgba('+ r +', '+ g +', '+ b +', 0.8)'
        r += 4;
        g += 2;
        b += 2;
    };


    var featuresChartData = {
      labels : [{% for f in features %}
                "{{f}}",
                {% endfor %}],
      datasets : [{
          backgroundColor: chartColors,
          borderColor: "rgba(91,98,65,1)",
          borderCapStyle: 'butt',
          borderJoinStyle: 'miter',
          pointBorderColor: "rgba(91,98,65,1)",
          pointBackgroundColor: "#fff",
          pointBorderWidth: 1,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "rgba(91,98,65,1)",
          pointHoverBorderColor: "rgba(220,220,220,1)",
          pointHoverBorderWidth: 2,
          data : [{% for v in values %}
                  {{v}},
                  {% endfor %}],
          spanGaps: false,

      }]
    }

    canvas.style.width='100%';
    canvas.width  = canvas.offsetWidth;

    var featuresChart{{ change }} = new Chart(ctx, {
      type: 'doughnut',
      data: featuresChartData,
      options: {
             legend: {
                display: false
             }
          }
    });
{% endfor %}
}


