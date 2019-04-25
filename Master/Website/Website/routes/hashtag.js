var express = require('express');
var router = express.Router();
 
router.get('/hashtag', function(req, res) {
	res.render('hashtag');
});
 
module.exports = router;