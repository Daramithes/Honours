$("#clicker").click(function() {
	username = $("#inputPlaceholderEx").val()
	$.ajax({
		  type:'get',
		  url:'http://localhost:5000/ExecuteTest/' + username,
		  cache:false,
		  async:'asynchronous',
		  dataType:'json',
		  success: function(data) {
			console.log(JSON.stringify(data))
		  },
		  error: function(request, status, error) {
			console.log("Error: " + error)
		  }
	   })
	console.log("Running: Anaylser");
	console.log("Running: Breakdown");
	console.log("Running: Visualizer");
	
});