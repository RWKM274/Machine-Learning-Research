
function changer(input){
	let baseImage; 
	if(input.files && input.files[0]){
		var reader = new FileReader();
		reader.onload = function(e){
			$('#heck')
			.attr('src', e.target.result)
			.width(300)
			.height(300);
			let dataUrl = reader.result;
			baseImage = dataUrl.replace("data:image/png;base64,","");
			console.log(baseImage);
		}
	};
	let message = {
		image: baseImage
	}
	console.log(message);
	$.post('/sample', JSON.stringify(message), function(response){
		answer(response);
	});
}

function loader(){
	$('#loading')
	.attr('src','https://hanslodge.com/images/BcgrpBzqi.png')
	.width(300)
	.height(300);
}

function answer(val){
	if(val == 1){
		$('#result')
		.attr('src', 'http://getdrawings.com/img/white-dog-silhouette-20.png')
		.width(300)
		.height(300);
	}else if(val == 2){
		$('#result')
		.attr('src','http://getdrawings.com/img/white-cat-silhouette-23.png')
		.width(300)
		.height(300);
	}else{
		$('#result').attr('src','')
	}
}