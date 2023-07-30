async function getRes() {
    await fetch('http://127.0.0.1:5000/park', {
        method: 'GET',
        headers: {
            'Accept': 'application/json',
        },
    })
        .then(response => response.json())
        .then(response => {
            const container = document.getElementById('container')
            container.innerHTML = ''
            total_free = 0
            total_occupied = 0
            final_div = ''
            for (let i = 0; i < response.length; i++) {
                free = response[i]['free']
                occupied = response[i]['occupied']
                total_free += free
                total_occupied += occupied
                if(i%2 == 0)
                    final_div += `<div class="w3-row-padding w3-padding-16 w3-center" id=container>`
                final_div += `<div class="w3-half">
                            <img src="${'data:image/png;base64,' + response[i]['frame']}" style="width:100%"">
                            <h3>Camera ${response[i]['number']}</h2>
                            <p>This part of the parking lot has <b>${response[i]['free']} free</b> slots and <b>${response[i]['occupied']} occupied</b> slots.</p>
                        </div>`
                if(i%2 == 1)
                    final_div += '</div>'
            }
            container.innerHTML = final_div
            const free_value = document.getElementById('free')
            const occupied_value = document.getElementById('occupied')
            free_value.innerHTML = `Total Free Slots: ${total_free}`
            occupied_value.innerHTML = `Total Occupied Slots: ${total_occupied}`
        })
}
setInterval(getRes, 5000)