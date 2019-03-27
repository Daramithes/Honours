$("#clicker").click(function() {
	GetManys()
	
});


function prepdata (data, caption, colours){
	var dataSource = {
	  "chart": {
		"caption": caption,
		"plottooltext": "<b>$percentValue</b> of Tweets have some relation to $label",
		"palettecolors": colours,
		"showlegend": "0",
		"showpercentvalues": "1",
		"legendposition": "bottom",
		"usedataplotcolorforlabels": "1",
		"theme": "fusion",
		"bgColor": "#000000"
	  },
	"data": data}
	  return dataSource
}; 

function CallTest(){
$.ajax({
		  type:'get',
		  url:'http://localhost:5000/testDump',
		  cache:false,
		  async:'asynchronous',
		  dataType:'text',
		  success: function(data) {
			test = JSON.parse(data)
			tests = prep(test)
			Colours = ColourConfiguration(test)
			dataSource = prepdata(tests, Colours)
			injectChart(dataSource)
			
			
		  },
		  error: function(request, status, error) {
			console.log("Error: " + error)
		  }
	   })

}
function injectChart(dataSource, renderLocation){
	FusionCharts.ready(function() {
	   var myChart = new FusionCharts({
		  type: "pie2d",
		  renderAt: renderLocation,
		  width: "100%",
		  height: "300",
		  dataFormat: "json",
		  dataSource
	   }).render();
	});
}

function prep(JSONobj){
	var array = []

	$.each(JSONobj, function (index, value) {
		data = {label: index, value: value}
		array.push(data)
	});
return array}

function ColourConfiguration(labels){
	Partys = {	  "Labour": "#dc241f",
                  "Conservative": "#0087dc",
                  "SNP": "#fff95d",
                  "Green": "#6ab023",
                  "UKIP": "#70147a",
                  "DUP":"#d46a4c",
                  "LibDem": "#fdbb30",
				  "Anti-Labour": "#dc241f",
                  "Anti-Conservative": "#0087dc",
                  "Anti-SNP": "#fff95d",
                  "Anti-Green": "#6ab023",
                  "Anti-UKIP": "#70147a",
                  "Anti-DUP":"#d46a4c",
                  "Anti-LibDem": "#fdbb30",
				  "Non-Aligned": "#ffffff",
				  "Republican": "#ff0000",
                  "Democratic": "#0015BC"}
	PartyColours = []
	$.each(labels, function (index, value) {
		PartyColours.push(Partys[(value["label"])])
	});
	
	return PartyColours
	
}

function GetMany(){
	$.ajax({
			  type:'get',
			  url:'http://localhost:5000/Cunt',
			  cache:false,
			  async:'asynchronous',
			  dataType:'text',
			  success: function(data) {
				x = data
			  },
			  error: function(request, status, error) {
				console.log("Error: " + error)
			  }
		   })	}

function ParseJSON(Collection){
	Collection = JSON.parse(Collection)
	British = Collection[0]
	American = Collection[1]
}

function InjectAll(Collection){
	ParseJSON(Collection)
	BritishInjection()
	AmericanInjection()}

function BritishInjection(){
	tester = []
		$.each(British, function(index,value) {
    tester.push(prep(value))})
	list = ["BritishChartOne", "BritishChartTwo"]
	ChartNames = ["British Breakdown Of Sentences", "British Breakdown Of Sentences w/ Sentiment"]
	$.each(tester, function(index,value) {
		Colours = ColourConfiguration(value)
		dataSource = prepdata(value, ChartNames[index], Colours)
		injectChart(dataSource, list[index])	
	});
}	

function AmericanInjection(){
	tester = []
		$.each(American, function(index,value) {
    tester.push(prep(value))})
	list = ["AmericanChartOne", "AmericanChartTwo"]
	ChartNames = ["American Breakdown Of Tweets", "American Breakdown Of Tweets w/ Sentiment"]
	$.each(tester, function(index,value) {
		Colours = ColourConfiguration(value)
		dataSource = prepdata(value, ChartNames[index], Colours)
		injectChart(dataSource, list[index])	
	});
}

function GetManys(){
	speech = $("#inputPlaceholderEx").val()
	$.ajax({
			  type:'get',
			  url:'http://localhost:5000/Text/' + speech,
			  cache:false,
			  async:'asynchronous',
			  dataType:'text',
			  success: function(data) {
				Results = data
				console.log("Response Successful")
				InjectAll(Results)
				$("#informationContainer").show()
			  },
			  error: function(request, status, error) {
				console.log("Error: " + error)
			  }
		   })	}

	