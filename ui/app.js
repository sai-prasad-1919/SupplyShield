document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const loadingEl = document.getElementById('loading');
    const resultsEl = document.getElementById('results');
    
    // Result elements
    const statusCard = document.getElementById('status-card');
    const statusHeading = document.getElementById('status-heading');
    const probVal = document.getElementById('prob-val');
    const statusIcon = document.getElementById('status-icon');
    
    const guidanceList = document.getElementById('guidance-list');
    const driversList = document.getElementById('drivers-list');
    const baseVal = document.getElementById('base-val');

    // Mappings for feature names
    const featureLabels = {
        'distance_km': 'Distance (km)',
        'weather': 'Weather Conditions',
        'traffic': 'Traffic Level',
        'vehicle_type': 'Vehicle Type'
    };

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading, hide results
        loadingEl.classList.remove('hidden');
        resultsEl.classList.add('hidden');
        
        // Get raw input values
        const rawDistance = parseFloat(document.getElementById('distance').value);
        const weather = parseFloat(document.getElementById('weather').value);
        const traffic = parseFloat(document.getElementById('traffic').value);
        const vehicle = parseFloat(document.getElementById('vehicle').value);
        
        // In a real application, the UI would send raw values to the backend,
        // and the backend would scale/encode them using the pre-fitted metadata.
        // For this demo, we apply a mock standard scaling to distance.
        // Mean ~ 10, std ~ 9 (from our exploration)
        const scaledDistance = (rawDistance - 10.12) / 9.39;
        
        // Order must match INPUT_DIM in the model:
        // [distance_km, weather, traffic, vehicle_type]
        const payload = {
            features: [scaledDistance, weather, traffic, vehicle]
        };

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();
            renderResults(data);
            
        } catch (error) {
            console.error('Error fetching prediction:', error);
            alert('Failed to connect to the SupplyShield API. Make sure the FastAPI server is running.');
        } finally {
            loadingEl.classList.add('hidden');
            resultsEl.classList.remove('hidden');
        }
    });

    function renderResults(data) {
        const pred = data.prediction;
        const expl = data.explainability;
        const guide = data.guidance;

        // 1. Update Status Card
        const probPct = (pred.delay_probability * 100).toFixed(1);
        probVal.textContent = `${probPct}%`;
        
        if (pred.is_delayed) {
            statusCard.classList.add('delayed');
            statusHeading.textContent = 'High Risk of Delay';
            probVal.style.color = 'var(--status-danger)';
            statusIcon.textContent = '⚠️';
        } else {
            statusCard.classList.remove('delayed');
            statusHeading.textContent = 'On Schedule';
            probVal.style.color = 'var(--status-success)';
            statusIcon.textContent = '✅';
        }

        // 2. Update AI Guidance
        guidanceList.innerHTML = '';
        guide.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            guidanceList.appendChild(li);
        });

        // 3. Update SHAP Drivers
        if (expl && expl.top_drivers) {
            baseVal.textContent = `${(expl.base_value * 100).toFixed(1)}%`;
            
            driversList.innerHTML = '';
            expl.top_drivers.forEach(driver => {
                const li = document.createElement('li');
                
                const isIncreases = driver.direction === 'increases_delay';
                li.className = `driver-item ${isIncreases ? '' : 'decreases'}`;
                
                const sign = isIncreases ? '+' : '';
                const impactColor = isIncreases ? 'var(--status-danger)' : 'var(--status-success)';
                
                const label = featureLabels[driver.feature] || driver.feature;
                
                li.innerHTML = `
                    <span class="driver-name">${label}</span>
                    <span class="driver-impact" style="color: ${impactColor}">${sign}${(driver.impact * 100).toFixed(1)}%</span>
                `;
                
                driversList.appendChild(li);
            });
        }
    }
});
