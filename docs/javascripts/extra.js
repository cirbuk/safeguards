document.addEventListener('DOMContentLoaded', function () {
    // Add GitHub repository details to the header
    setTimeout(function () {
        // Note: We're no longer manually handling repository statistics
        // as they are now managed by Material for MkDocs theme configuration

        // Hide tabs container on initial load
        const tabsContainer = document.querySelector('.md-tabs');
        if (tabsContainer) {
            tabsContainer.style.display = 'none';
        }

        // Create custom top navigation
        const header = document.querySelector('.md-header__inner');
        const title = document.querySelector('.md-header__title');

        if (header && title) {
            const nav = document.createElement('div');
            nav.classList.add('md-header__nav');

            // Define navigation items matching OpenAI's layout
            const navItems = [
                { text: 'Home', url: '.' },
                { text: 'Getting Started', url: 'quickstart/' },
                { text: 'Guides', url: 'guides/' },
                { text: 'API Reference', url: 'api/core/' },
                { text: 'Examples', url: 'examples/' }
            ];

            // Create navigation links
            navItems.forEach(item => {
                const link = document.createElement('a');
                link.classList.add('md-header__nav-link');
                link.textContent = item.text;
                link.href = item.url;

                // Check if current page is in this section to add active state
                const path = window.location.pathname;
                if ((item.url !== '.' && path.includes(item.url)) ||
                    (item.url === '.' && path.endsWith('/') || path.endsWith('/index.html'))) {
                    link.classList.add('md-header__nav-link--active');
                }

                nav.appendChild(link);
            });

            // Insert after title
            header.insertBefore(nav, title.nextSibling);
        }

        // Enhance the sidebar navigation
        enhanceSidebar();
    }, 100);

    // Function to enhance the sidebar navigation to match OpenAI's style
    function enhanceSidebar () {
        // Add styles to active navigation items
        const activeItems = document.querySelectorAll('.md-nav__item--active');
        activeItems.forEach(item => {
            const link = item.querySelector('.md-nav__link');
            if (link) {
                link.classList.add('md-nav__link--active');
            }
        });

        // Add Table of Contents title to match OpenAI's site
        const tocSidebar = document.querySelector('.md-sidebar--secondary');
        if (tocSidebar) {
            const tocNav = tocSidebar.querySelector('.md-nav');
            if (tocNav) {
                const tocTitle = document.createElement('div');
                tocTitle.classList.add('md-nav__title');
                tocTitle.textContent = 'Table of contents';
                tocNav.insertBefore(tocTitle, tocNav.firstChild);
            }
        }
    }

    // Add scroll event listener to show tabs when scrolling up
    let lastScrollTop = 0;
    window.addEventListener('scroll', function () {
        const tabsContainer = document.querySelector('.md-tabs');
        if (!tabsContainer) return;

        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

        // When scrolling up and not at the top
        if (scrollTop < lastScrollTop && scrollTop > 60) {
            tabsContainer.style.display = 'block';
        } else {
            tabsContainer.style.display = 'none';
        }

        lastScrollTop = scrollTop;
    });
});
