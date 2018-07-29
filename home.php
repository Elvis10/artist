<head>
	<h1>BLOCKCHAIN</h1>
	<h4>Ejemplo de prueba</h4>
</head>
<body>
	<h4>Gracias por jugar con nosotros</h4>
	<?php
		$top = $_POST['top'];
		$moodEncoded = $_POST['moodEncoded'];
		$tempoEncoded = $_POST['tempoEncoded'];
		$genreEncoded = $_POST['genreEncoded'];
		$artist_typeEncoded = $_POST['artist_typeEncoded'];
		$edadEncoded = $_POST['edadEncoded'];
		$durationEncoded = $_POST['durationEncoded'];

		$data = array($top,$moodEncoded,$tempoEncoded,$genreEncoded,$artist_typeEncoded,$edadEncoded,$durationEncoded);
		//$resultado=shell_exec("python3 /var/www/html/motorpwm.py " . escapeshellarg(json_encode($data)));
		//$resultado = `python Final3.py` . escapeshellarg(json_encode($data));
		//exec("python contar.py '".$data."'",$resultData);
		//https://es.stackoverflow.com/questions/106407/como-pasar-variables-de-php-a-python-y-viceversa-mediante-json
		//https://stackoverflow.com/questions/28778384/sending-json-data-from-python-to-php?newreg=971e318645554b1bba64e1a7f65645fa
		$resultado = shell_exec("python Final3.py " . escapeshellarg(json_encode($data)))

		$resultData = json_decode($resultado, true);
		var_dump($resultData);

	?>
</body>