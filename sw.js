
const KERNEL = 'micro-asxr-kernel-v1';
self.addEventListener('install', e=>{
  e.waitUntil((async()=>{
    const c = await caches.open(KERNEL);
    try {
      for (const path of ['/index.html','/atomic.css','/MICRO.ASXR']) {
        const res = await fetch(path);
        if (res.ok) await c.put(path, res.clone());
      }
    } catch(_) {}
    self.skipWaiting();
  })());
});
self.addEventListener('activate', e=>{ e.waitUntil(self.clients.claim()); });

function isRoute(url){
  try {
    const u = new URL(url);
    if (u.origin !== self.location.origin) return false;
    return !u.pathname.match(/\.(js|css|json|png|jpg|svg|ico|webmanifest|map)$/);
  } catch { return false; }
}

self.addEventListener('fetch', e=>{
  const url = new URL(e.request.url);
  e.respondWith((async()=>{
    const c = await caches.open(KERNEL);
    if (url.origin === self.location.origin &&
        (url.pathname==='/index.html' || url.pathname==='/MICRO.ASXR' || url.pathname==='/atomic.css')) {
      const hit = await c.match(url.pathname);
      return hit || fetch(e.request);
    }
    if (url.origin === self.location.origin && isRoute(url.href)) {
      const shell = await c.match('/index.html');
      return shell || fetch('/index.html');
    }
    try {
      const fresh = await fetch(e.request);
      c.put(e.request, fresh.clone());
      return fresh;
    } catch(_) {
      const hit = await c.match(e.request);
      if (hit) return hit;
      if (url.origin === self.location.origin) {
        const shell = await c.match('/index.html');
        if (shell) return shell;
      }
      throw _;
    }
  })());
});
