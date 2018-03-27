// Global parameters:
// do not resize the chart canvas when its container does (keep at 600x400px)
Chart.defaults.global.responsive = false;
// define the chart data
window.onload = function() {

{% set labels, values, tweets = graph_data %}

var tweets_per_date = {
    {% for date, tweet in tweets.items() %}
        "{{date}}": {{tweet|tojson|safe}},
    {% endfor %}
}


var chartData = {
  labels : [{% for d in labels %}
            "{{d}}",
            {% endfor %}],
  datasets : [{
      fill: true,
      lineTension: 0.1,
      backgroundColor: "rgba(75,192,83,0.4)",
      borderColor: "rgba(75,192,83,1)",
      borderCapStyle: 'butt',
      borderJoinStyle: 'miter',
      pointBorderColor: "rgba(75,192,192,1)",
      pointBackgroundColor: "#fff",
      pointBorderWidth: 1,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "rgba(75,192,83,1)",
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

// get chart canvas
var canvas = document.getElementById("currencyChart");

canvas.style.width='100%';
canvas.width  = canvas.offsetWidth;

// create the chart using the chart canvas
var currencyChart = new Chart(canvas.getContext("2d"), {
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


{% set features_with_values = features_data["Down"] %}
{% set features, values = features_with_values|unzip %}

var featuresChartData = {
  labels : [{% for f in features %}
            "{{f}}",
            {% endfor %}],
  datasets : [{
      backgroundColor: "rgba(75,192,83,0.4)",
      borderColor: "rgba(75,192,83,1)",
      borderCapStyle: 'butt',
      borderJoinStyle: 'miter',
      pointBorderColor: "rgba(75,192,192,1)",
      pointBackgroundColor: "#fff",
      pointBorderWidth: 1,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "rgba(75,192,83,1)",
      pointHoverBorderColor: "rgba(220,220,220,1)",
      pointHoverBorderWidth: 2,
      data : [{% for v in values %}
              {{v}},
              {% endfor %}],
      spanGaps: false,

  }]
}

var canvas = document.getElementById("featuresChart");

canvas.style.width='100%';
canvas.width  = canvas.offsetWidth;

var featuresChart = new Chart(canvas.getContext("2d"), {
  type: 'pie',
  data: featuresChartData,
  options: {
         legend: {
            display: false
         }
      }
});

}
