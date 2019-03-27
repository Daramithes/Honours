var express = require('express');
var router = express.Router();
 
router.get('/speech', function(req, res) {
	res.render('speech');
});
 
module.exports = router;