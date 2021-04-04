module.exports = {
  title: 'SuSi',
  base: "/susi/",
  description: "Susi is a Python package for unsupervised, supervised and semi-supervised self-organizing maps (SOM).",
  head: [
    ['link', { rel: "apple-touch-icon", sizes: "180x180", href: "/assets/favicons/apple-icon.png"}],
    ['link', { rel: "icon", type: "image/png", sizes: "32x32", href: "/assets/favicons/favicon-32x32.png"}],
    ['link', { rel: "icon", type: "image/png", sizes: "16x16", href: "/assets/favicons/favicon-16x16.png"}],
    ['link', { rel: "manifest", href: "/assets/favicons/manifest.json"}],
    ['link', { rel: "shortcut icon", href: "/assets/favicons/favicon.ico"}],
    ['meta', { name: "msapplication-TileColor", content: "#ffffff"}],
    ['meta', { name: "msapplication-TileImage", content: "/assets/favicons/ms-icon-144x144.png"}],
    ['meta', { name: "msapplication-config", content: "/assets/favicons/browserconfig.xml"}],
    ['meta', { name: "theme-color", content: "#ffffff"}],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }]
  ],
  themeConfig: {
    smoothScroll: true,
    repo: 'https://github.com/felixriese/susi',
    logo: 'https://github.com/felixriese/susi/blob/main/docs/_static/susi_logo_small.png?raw=true',
    nav: [{text: 'Documentation', link: 'https://susi.readthedocs.io/en/latest/?badge=latest' }],
  },
}
