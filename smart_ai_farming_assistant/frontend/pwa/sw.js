// Service Worker for Smart AI Farming Assistant
// Provides offline caching for static assets

const CACHE_NAME = 'farming-assistant-v1';
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/styles.css',
    '/app.js',
    '/api.js',
    '/i18n/en.json',
    '/i18n/hi.json',
    '/i18n/gu.json',
    '/pwa/manifest.json'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('Service Worker installing...');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => {
                console.log('Static assets cached successfully');
                return self.skipWaiting();
            })
            .catch((error) => {
                console.error('Failed to cache static assets:', error);
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('Service Worker activating...');
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => {
                        if (cacheName !== CACHE_NAME) {
                            console.log('Deleting old cache:', cacheName);
                            return caches.delete(cacheName);
                        }
                    })
                );
            })
            .then(() => {
                console.log('Service Worker activated');
                return self.clients.claim();
            })
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Only handle GET requests
    if (request.method !== 'GET') {
        return;
    }

    // Handle API requests - network first, no caching
    if (url.pathname.startsWith('/predict/') || 
        url.pathname.startsWith('/recommend/') || 
        url.pathname.startsWith('/schemes/') || 
        url.pathname.startsWith('/disease/') || 
        url.pathname.startsWith('/chat') ||
        url.pathname === '/health') {
        
        event.respondWith(
            fetch(request)
                .catch(() => {
                    // Return a basic offline response for API calls
                    return new Response(
                        JSON.stringify({ 
                            error: 'Offline - API not available',
                            offline: true 
                        }),
                        {
                            status: 503,
                            headers: { 'Content-Type': 'application/json' }
                        }
                    );
                })
        );
        return;
    }

    // Handle static assets - cache first, fallback to network
    event.respondWith(
        caches.match(request)
            .then((cachedResponse) => {
                if (cachedResponse) {
                    console.log('Serving from cache:', request.url);
                    return cachedResponse;
                }

                console.log('Fetching from network:', request.url);
                return fetch(request)
                    .then((networkResponse) => {
                        // Cache successful responses for static assets
                        if (networkResponse.status === 200) {
                            const responseClone = networkResponse.clone();
                            caches.open(CACHE_NAME)
                                .then((cache) => {
                                    cache.put(request, responseClone);
                                });
                        }
                        return networkResponse;
                    })
                    .catch((error) => {
                        console.error('Network request failed:', error);
                        
                        // Return a basic offline page for navigation requests
                        if (request.mode === 'navigate') {
                            return caches.match('/index.html');
                        }
                        
                        throw error;
                    });
            })
    );
});

// Handle messages from the main thread
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});