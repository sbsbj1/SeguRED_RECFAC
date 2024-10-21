var map = L.map('map').setView([-33.4489, -70.6693], 13);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Cargar los datos de paradas
fetch('paradas.json')
    .then(response => response.json())
    .then(data => {
        data.forEach(parada => {
            // Crear un marcador para cada parada
            const marker = L.marker([parada.lat_parada, parada.lon_parada]).addTo(map);
            // Agregar un popup con información
            marker.bindPopup(`<strong>${parada.nombre_parada}</strong><br>Evasiones: ${parada.evasiones}`);
        });
    })
    .catch(error => console.error('Error al cargar las paradas:', error));

