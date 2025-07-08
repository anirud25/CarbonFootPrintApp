document.getElementById('carbonForm').addEventListener('submit', async function (event) {
    event.preventDefault(); // Prevent form refresh

    const energySource = document.getElementById('energySource').value;
    const dietType = document.getElementById('dietType').value;
    const transportMode = document.getElementById('transportMode').value;
    const electricityUsage = document.getElementById('electricityUsage').value;
    const plasticUsage = document.getElementById('plasticUsage').value;
    const disposalMethod = document.getElementById('disposalMethod').value

    const inputData = {
        EnergySource: parseInt(energySource),
        DietType: parseInt(dietType),
        TransportationMode: parseInt(transportMode),
        MonthlyElectricityConsumption: parseFloat(electricityUsage),
        PlasticUsage: parseInt(plasticUsage),
        DisposalMethod: parseInt(disposalMethod)
    };


    try {
        // Make the POST request to the predict API hosted in another bucket
        const response = await fetch('http://127.0.0.1:80/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(inputData),
        });

        const data = await response.json();
        document.getElementById('carbonOutput').textContent = `Your estimated CO2 emission is: ${data.prediction} kg CO2`;

        // Make another request to get recommendations
        const recResponse = await fetch('http://127.0.0.1:80/recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(inputData),
        });

        const recData = await recResponse.json();
        const recommendationsOutputElement = document.getElementById('recommendationsOutput');
        if (recData.recommendations && recData.recommendations.trim() !== "") {
            // Split recommendations into an array using ';'
            const recommendations = recData.recommendations.split(';');
        
            // Create a styled unordered list for the recommendations
            let outputHtml = "<ul>";
            recommendations.forEach((rec) => {
                outputHtml += `<li>${rec.trim()}</li>`; // Trim whitespace and add as a list item
            });
            outputHtml += "</ul>";
        
            recommendationsOutputElement.innerHTML = outputHtml; // Update the output element with the list
        } else {
            // Display "You are good to go!" if no recommendations are provided
            recommendationsOutputElement.textContent = "You are good to go!";
            recommendationsOutputElement.style.color = "green"; // Optional: Add styling for the message
            recommendationsOutputElement.style.textAlign = "center"; // Center horizontally
            recommendationsOutputElement.style.marginTop = "20px"; // Add spacing
            recommendationsOutputElement.style.fontWeight = "bold"; // Make it bold
        }
        

        // Display the results box
        document.getElementById('resultBox').classList.remove('hidden');
    } catch (error) {
        console.error('Error:', error);
    }
});
