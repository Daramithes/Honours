var express = require('express');
var routes = require('./routes/index');
var speech = require('./routes/speech');
var hashtag = require('./routes/hashtag');
var cookieParser = require('cookie-parser');
var app = express();
 
app.set('views', __dirname + '/views');
app.set('view engine', 'ejs');
 
app.use(cookieParser());
app.use(express.static(__dirname + '/public'));
 
app.use('/', routes);
app.use('/', speech);
app.use('/', hashtag);

app.use(function(req, res, next) {
	var err = new Error('Not Found');
	err.status = 404;
	next(err);
});
 
app.use(function(err, req, res, next) {
		res.status(err.status || 500);
		res.render('error', {
		message: err.message,
		error: err
	});
});
 
app.listen(8000);