document.addEventListener('DOMContentLoaded', function () {
    // Add GitHub repository details to the header
    setTimeout(function () {
        const sourceElement = document.querySelector('.md-header__source');

        if (sourceElement) {
            // Completely replace the repository display to match OpenAI style
            const sourceContent = sourceElement.querySelector('.md-source');

            if (sourceContent) {
                // Store the original href
                const repoUrl = sourceContent.getAttribute('href');

                // Create a new source element
                const newSource = document.createElement('a');
                newSource.classList.add('md-source');
                newSource.href = repoUrl;

                // Create the GitHub icon
                const iconWrapper = document.createElement('div');
                iconWrapper.classList.add('md-source__icon');

                const githubIcon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                githubIcon.setAttribute('viewBox', '0 0 16 16');
                githubIcon.setAttribute('width', '16');
                githubIcon.setAttribute('height', '16');

                const iconPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                iconPath.setAttribute('d', 'M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z');

                githubIcon.appendChild(iconPath);
                iconWrapper.appendChild(githubIcon);

                // Create the repository info
                const repoWrapper = document.createElement('div');
                repoWrapper.classList.add('md-source__repository');

                // Create facts list
                const factsList = document.createElement('ul');
                factsList.classList.add('md-source__facts');

                // Version fact
                const versionFact = document.createElement('li');
                versionFact.classList.add('md-source__fact', 'md-source__fact--version');
                versionFact.textContent = 'v0.1.0';
                factsList.appendChild(versionFact);

                // Stars fact
                const starsFact = document.createElement('li');
                starsFact.classList.add('md-source__fact', 'md-source__fact--stars');
                starsFact.textContent = '0';
                factsList.appendChild(starsFact);

                // Forks fact
                const forksFact = document.createElement('li');
                forksFact.classList.add('md-source__fact', 'md-source__fact--forks');
                forksFact.textContent = '0';
                factsList.appendChild(forksFact);

                // Assemble the elements
                repoWrapper.appendChild(factsList);
                newSource.appendChild(iconWrapper);
                newSource.appendChild(repoWrapper);

                // Replace the original content
                sourceElement.innerHTML = '';
                sourceElement.appendChild(newSource);
            }
        }

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
