// Enhanced Smart AI Farming Assistant - Main Application JavaScript

class FarmingApp {
    constructor() {
        this.currentLang = 'en';
        this.translations = {};
        this.isOnline = navigator.onLine;
        this.retryAttempts = 3;
        this.init();
    }

    async init() {
        await this.loadTranslations();
        this.setupEventListeners();
        this.setupTabs();
        this.setupNetworkMonitoring();
        this.setupFormValidation();
        this.updateLanguage();
        this.showWelcomeMessage();
    }

    async loadTranslations() {
        try {
            const languages = ['en', 'hi', 'gu'];
            const promises = languages.map(async lang => {
                try {
                    const response = await fetch(`i18n/${lang}.json`);
                    if (response.ok) {
                        this.translations[lang] = await response.json();
                    }
                } catch (error) {
                    console.warn(`Failed to load ${lang} translations:`, error);
                }
            });
            await Promise.all(promises);
        } catch (error) {
            console.error('Failed to load translations:', error);
        }
    }

    setupNetworkMonitoring() {
        window.addEventListener('online', () => {
            this.isOnline = true;
            this.showNotification('Connection restored', 'success');
        });

        window.addEventListener('offline', () => {
            this.isOnline = false;
            this.showNotification('Working offline', 'warning');
        });
    }

    setupFormValidation() {
        // Add real-time validation for numeric inputs
        const numericInputs = document.querySelectorAll('input[type="number"]');
        numericInputs.forEach(input => {
            input.addEventListener('input', this.validateNumericInput.bind(this));
            input.addEventListener('blur', this.validateNumericInput.bind(this));
        });

        // Add validation for coordinates
        const latInput = document.getElementById('lat');
        const lonInput = document.getElementById('lon');
        
        if (latInput) {
            latInput.addEventListener('input', () => this.validateCoordinate(latInput, -90, 90));
        }
        if (lonInput) {
            lonInput.addEventListener('input', () => this.validateCoordinate(lonInput, -180, 180));
        }
    }

    validateNumericInput(event) {
        const input = event.target;
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);

        input.classList.remove('error', 'success');

        if (input.value && !isNaN(value)) {
            if ((min !== undefined && value < min) || (max !== undefined && value > max)) {
                input.classList.add('error');
                this.showInputError(input, `Value must be between ${min || 'any'} and ${max || 'any'}`);
            } else {
                input.classList.add('success');
                this.hideInputError(input);
            }
        }
    }

    validateCoordinate(input, min, max) {
        const value = parseFloat(input.value);
        input.classList.remove('error', 'success');

        if (input.value && !isNaN(value)) {
            if (value < min || value > max) {
                input.classList.add('error');
                this.showInputError(input, `${input.id === 'lat' ? 'Latitude' : 'Longitude'} must be between ${min} and ${max}`);
            } else {
                input.classList.add('success');
                this.hideInputError(input);
            }
        }
    }

    showInputError(input, message) {
        let errorDiv = input.parentNode.querySelector('.input-error');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.className = 'input-error';
            input.parentNode.appendChild(errorDiv);
        }
        errorDiv.textContent = message;
    }

    hideInputError(input) {
        const errorDiv = input.parentNode.querySelector('.input-error');
        if (errorDiv) {
            errorDiv.remove();
        }
    }

    setupEventListeners() {
        // Language selector with smooth transition
        document.getElementById('languageSelect').addEventListener('change', (e) => {
            this.currentLang = e.target.value;
            this.updateLanguage();
            this.showNotification('Language updated', 'success');
        });

        // Enhanced form submissions with loading states
        this.setupFormHandler('weatherForm', this.handleWeatherPrediction.bind(this));
        this.setupFormHandler('cropForm', this.handleCropRecommendation.bind(this));
        this.setupFormHandler('diseaseForm', this.handleDiseaseDiganosis.bind(this));
        this.setupFormHandler('schemesForm', this.handleSchemeSearch.bind(this));
        this.setupFormHandler('chatForm', this.handleChat.bind(this));

        // Add geolocation button
        this.addGeolocationButton();
    }

    setupFormHandler(formId, handler) {
        const form = document.getElementById(formId);
        if (form) {
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                if (this.validateForm(form)) {
                    await handler();
                }
            });
        }
    }

    validateForm(form) {
        const requiredInputs = form.querySelectorAll('[required]');
        let isValid = true;

        requiredInputs.forEach(input => {
            if (!input.value.trim()) {
                input.classList.add('error');
                this.showInputError(input, 'This field is required');
                isValid = false;
            } else {
                input.classList.remove('error');
                this.hideInputError(input);
            }
        });

        return isValid;
    }

    addGeolocationButton() {
        const latInput = document.getElementById('lat');
        if (latInput && 'geolocation' in navigator) {
            const geoButton = document.createElement('button');
            geoButton.type = 'button';
            geoButton.className = 'geo-button';
            geoButton.innerHTML = '📍 Use My Location';
            geoButton.onclick = this.getCurrentLocation.bind(this);
            
            latInput.parentNode.appendChild(geoButton);
        }
    }

    async getCurrentLocation() {
        if (!navigator.geolocation) {
            this.showNotification('Geolocation not supported', 'error');
            return;
        }

        const geoButton = document.querySelector('.geo-button');
        if (geoButton) {
            geoButton.classList.add('loading');
            geoButton.disabled = true;
        }

        try {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 300000
                });
            });

            document.getElementById('lat').value = position.coords.latitude.toFixed(4);
            document.getElementById('lon').value = position.coords.longitude.toFixed(4);
            this.showNotification('Location updated', 'success');
        } catch (error) {
            this.showNotification('Failed to get location', 'error');
        } finally {
            if (geoButton) {
                geoButton.classList.remove('loading');
                geoButton.disabled = false;
            }
        }
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;

                // Smooth transition
                tabContents.forEach(content => {
                    if (content.classList.contains('active')) {
                        content.style.opacity = '0';
                        setTimeout(() => {
                            content.classList.remove('active');
                            content.style.opacity = '1';
                        }, 150);
                    }
                });

                setTimeout(() => {
                    tabButtons.forEach(btn => btn.classList.remove('active'));
                    button.classList.add('active');
                    document.getElementById(targetTab).classList.add('active');
                }, 150);
            });
        });
    }

    updateLanguage() {
        const elements = document.querySelectorAll('[data-i18n]');
        elements.forEach(element => {
            const key = element.dataset.i18n;
            const translation = this.translations[this.currentLang]?.[key] || key;
            
            // Smooth text transition
            element.style.opacity = '0.7';
            setTimeout(() => {
                element.textContent = translation;
                element.style.opacity = '1';
            }, 100);
        });
    }

    showWelcomeMessage() {
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages && chatMessages.children.length === 0) {
            this.addChatMessage('assistant', 'Welcome! I can help you with weather predictions, crop recommendations, disease diagnosis, and government schemes. How can I assist you today?');
        }
    }

    showResult(containerId, content, type = 'success') {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.style.opacity = '0';
        container.innerHTML = content;
        container.className = `result show ${type} fade-in`;
        
        setTimeout(() => {
            container.style.opacity = '1';
        }, 50);
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => notification.classList.add('show'), 100);
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    showLoading(formId) {
        const form = document.getElementById(formId);
        const button = form?.querySelector('button[type="submit"]');
        if (button) {
            button.classList.add('loading');
            button.disabled = true;
            button.dataset.originalText = button.textContent;
            button.textContent = 'Processing...';
        }
    }

    hideLoading(formId) {
        const form = document.getElementById(formId);
        const button = form?.querySelector('button[type="submit"]');
        if (button) {
            button.classList.remove('loading');
            button.disabled = false;
            button.textContent = button.dataset.originalText || button.textContent;
        }
    }

    async makeRequest(apiCall, retries = this.retryAttempts) {
        try {
            return await apiCall();
        } catch (error) {
            if (retries > 0 && (error.message.includes('fetch') || error.message.includes('network'))) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                return this.makeRequest(apiCall, retries - 1);
            }
            throw error;
        }
    }

    async handleWeatherPrediction() {
        this.showLoading('weatherForm');

        try {
            const formData = {
                lat: parseFloat(document.getElementById('lat').value),
                lon: parseFloat(document.getElementById('lon').value),
                month: parseInt(document.getElementById('month').value),
                humidity: document.getElementById('humidity').value ? parseFloat(document.getElementById('humidity').value) : null,
                cloud_cover: document.getElementById('cloudCover').value ? parseFloat(document.getElementById('cloudCover').value) : null,
                wind_kph: document.getElementById('windSpeed').value ? parseFloat(document.getElementById('windSpeed').value) : null,
                sun_hours: document.getElementById('sunHours').value ? parseFloat(document.getElementById('sunHours').value) : null,
                offline_only: document.getElementById('offlineOnly')?.checked || !this.isOnline
            };

            const result = await this.makeRequest(() => api.predictWeather(formData));

            const content = `
                <h3>🌤️ Weather Prediction</h3>
                <div class="prediction-grid">
                    <div class="prediction-item">
                        <span class="icon">🌧️</span>
                        <span class="label">Rainfall</span>
                        <span class="value">${result.rain_mm} mm</span>
                    </div>
                    <div class="prediction-item">
                        <span class="icon">🌡️</span>
                        <span class="label">Temperature</span>
                        <span class="value">${result.temp_c}°C</span>
                    </div>
                </div>
                <div class="why">${result.why}</div>
            `;

            this.showResult('weatherResult', content, 'success');
        } catch (error) {
            this.showResult('weatherResult', `<p>❌ Error: ${error.message}</p>`, 'error');
        } finally {
            this.hideLoading('weatherForm');
        }
    }

    async handleCropRecommendation() {
        this.showLoading('cropForm');

        try {
            const formData = {
                soil_type: document.getElementById('soilType').value,
                season: document.getElementById('season').value
            };

            const result = await this.makeRequest(() => api.recommendCrops(formData));

            const content = `
                <h3>🌾 Recommended Crops</h3>
                <div class="soil-info">
                    <p><strong>🏔️ Soil Type:</strong> ${result.soil_name}</p>
                    <p><strong>📋 Characteristics:</strong> ${result.characteristics}</p>
                </div>
                <div class="crops-grid">
                    ${result.crops.map(crop => `<div class="crop-item">🌱 ${crop}</div>`).join('')}
                </div>
                <div class="why">${result.why}</div>
            `;

            this.showResult('cropResult', content, 'success');
        } catch (error) {
            this.showResult('cropResult', `<p>❌ Error: ${error.message}</p>`, 'error');
        } finally {
            this.hideLoading('cropForm');
        }
    }

    async handleDiseaseDiganosis() {
        this.showLoading('diseaseForm');

        try {
            const formData = {
                text: document.getElementById('symptoms').value
            };

            const result = await this.makeRequest(() => api.diagnoseDiseae(formData));

            const content = `
                <h3>🔬 Diagnosis Result</h3>
                <div class="diagnosis-card">
                    <div class="diagnosis-item">
                        <span class="icon">🦠</span>
                        <span class="label">Likely Issue</span>
                        <span class="value">${result.likely}</span>
                    </div>
                    <div class="advice-section">
                        <h4>💡 Recommended Action</h4>
                        <p>${result.advice}</p>
                    </div>
                </div>
                <div class="why">${result.why}</div>
            `;

            this.showResult('diseaseResult', content, 'success');
        } catch (error) {
            this.showResult('diseaseResult', `<p>❌ Error: ${error.message}</p>`, 'error');
        } finally {
            this.hideLoading('diseaseForm');
        }
    }

    async handleSchemeSearch() {
        this.showLoading('schemesForm');

        try {
            const formData = {
                query: document.getElementById('schemeQuery').value || null,
                state: document.getElementById('schemeState').value || null
            };

            const result = await this.makeRequest(() => api.searchSchemes(formData));

            let content = '<h3>🏛️ Government Schemes</h3>';
            
            if (result.results.length === 0) {
                content += '<div class="no-results">📭 No schemes found matching your criteria. Try different search terms.</div>';
            } else {
                content += `<div class="schemes-count">Found ${result.results.length} scheme(s)</div>`;
                content += result.results.map(scheme => `
                    <div class="scheme-item bounce-in">
                        <h4>${scheme.name}</h4>
                        <div class="category">${scheme.category.replace('_', ' ')}</div>
                        <p>${scheme.description}</p>
                        <div class="scheme-details">
                            <p><strong>✅ Eligibility:</strong> ${scheme.eligibility}</p>
                            <div class="state">📍 ${scheme.state === 'all' ? 'All States' : scheme.state.toUpperCase()}</div>
                        </div>
                    </div>
                `).join('');
            }

            this.showResult('schemesResult', content, 'success');
        } catch (error) {
            this.showResult('schemesResult', `<p>❌ Error: ${error.message}</p>`, 'error');
        } finally {
            this.hideLoading('schemesForm');
        }
    }

    async handleChat() {
        const input = document.getElementById('chatInput');
        const messagesContainer = document.getElementById('chatMessages');
        const question = input.value.trim();

        if (!question) return;

        // Add user message with animation
        this.addChatMessage('user', question);
        input.value = '';

        // Show typing indicator
        const typingDiv = this.addTypingIndicator();

        try {
            const formData = {
                question: question,
                lang: this.currentLang
            };

            const result = await this.makeRequest(() => api.chat(formData));
            
            // Remove typing indicator
            typingDiv.remove();
            
            this.addChatMessage('assistant', result.response);
        } catch (error) {
            typingDiv.remove();
            this.addChatMessage('assistant', `❌ Error: ${error.message}`);
        }

        // Smooth scroll to bottom
        this.scrollChatToBottom();
    }

    addTypingIndicator() {
        const messagesContainer = document.getElementById('chatMessages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message assistant typing';
        typingDiv.innerHTML = `
            <div class="sender">Assistant</div>
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        `;
        messagesContainer.appendChild(typingDiv);
        this.scrollChatToBottom();
        return typingDiv;
    }

    addChatMessage(sender, message) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;
        
        const senderLabel = sender === 'user' ? '👤 You' : '🤖 Assistant';
        messageDiv.innerHTML = `
            <div class="sender">${senderLabel}</div>
            <div class="message">${message}</div>
            <div class="timestamp">${new Date().toLocaleTimeString()}</div>
        `;

        messagesContainer.appendChild(messageDiv);
        
        // Animate message appearance
        setTimeout(() => messageDiv.classList.add('show'), 50);
    }

    scrollChatToBottom() {
        const messagesContainer = document.getElementById('chatMessages');
        messagesContainer.scrollTo({
            top: messagesContainer.scrollHeight,
            behavior: 'smooth'
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.farmingApp = new FarmingApp();
});

// Add enhanced styles for new features
const enhancedStyles = `
<style>
.input-error {
    color: #f44336;
    font-size: 0.8rem;
    margin-top: 0.25rem;
}

.form-group input.error,
.form-group select.error,
.form-group textarea.error {
    border-color: #f44336;
    box-shadow: 0 0 0 3px rgba(244, 67, 54, 0.1);
}

.form-group input.success,
.form-group select.success,
.form-group textarea.success {
    border-color: var(--light-green);
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
}

.geo-button {
    margin-top: 0.5rem;
    padding: 0.5rem 1rem;
    background: var(--accent-green);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: var(--transition);
}

.geo-button:hover {
    background: var(--light-green);
    transform: translateY(-1px);
}

.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    color: white;
    font-weight: 500;
    z-index: 1000;
    transform: translateX(100%);
    transition: transform 0.3s ease;
}

.notification.show {
    transform: translateX(0);
}

.notification.success { background: var(--light-green); }
.notification.error { background: #f44336; }
.notification.warning { background: #ff9800; }
.notification.info { background: #2196f3; }

.prediction-grid, .crops-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.prediction-item, .crop-item {
    background: rgba(255, 255, 255, 0.8);
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    border: 1px solid rgba(76, 175, 80, 0.2);
}

.prediction-item .icon {
    font-size: 1.5rem;
    display: block;
    margin-bottom: 0.5rem;
}

.prediction-item .value {
    font-size: 1.2rem;
    font-weight: bold;
    color: var(--primary-green);
}

.diagnosis-card {
    background: rgba(255, 255, 255, 0.8);
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.diagnosis-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.advice-section {
    background: rgba(76, 175, 80, 0.1);
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid var(--light-green);
}

.schemes-count {
    background: var(--light-green);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}

.no-results {
    text-align: center;
    padding: 2rem;
    color: #666;
    font-style: italic;
}

.chat-message {
    opacity: 0;
    transform: translateY(10px);
    transition: var(--transition);
}

.chat-message.show {
    opacity: 1;
    transform: translateY(0);
}

.timestamp {
    font-size: 0.7rem;
    color: #999;
    margin-top: 0.25rem;
}

.typing-dots {
    display: flex;
    gap: 0.25rem;
    align-items: center;
}

.typing-dots span {
    width: 6px;
    height: 6px;
    background: var(--light-green);
    border-radius: 50%;
    animation: typing 1.4s infinite;
}

.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-10px); }
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', enhancedStyles);
