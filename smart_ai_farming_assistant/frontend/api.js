// API utility functions for Smart AI Farming Assistant

class API {
    constructor() {
        this.baseURL = '';
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }

    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    // Health check
    async checkHealth() {
        return this.get('/health');
    }

    // Weather prediction
    async predictWeather(data) {
        return this.post('/predict/weather', data);
    }

    // Crop recommendation
    async recommendCrops(data) {
        return this.post('/recommend/crops', data);
    }

    // Scheme search
    async searchSchemes(data) {
        return this.post('/schemes/search', data);
    }

    // Disease diagnosis
    async diagnoseDiseae(data) {
        return this.post('/disease/quick-diagnose', data);
    }

    // Chat
    async chat(data) {
        return this.post('/chat', data);
    }
}

// Create global API instance
window.api = new API();

// Gemini API configuration - using the provided API key
// const GEMINI (This line was incomplete and commented out to prevent syntax errors)
