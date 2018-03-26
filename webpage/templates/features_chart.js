Chart.defaults.global.responsive = false;

window.onload = function() {

{% set features_with_values = features_data["Down"] %}
{% set features, values = *features_with_values|zip %}

var chartData = {
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

var ctx = canvas.getContext("2d");

var featuresChart = new Chart(ctx, {
  type: 'pie',
  data: chartData,
  options: {
         legend: {
            display: true
         }
      }
});
}
