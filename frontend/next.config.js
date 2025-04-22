/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    swcMinify: true,
    env: {
        NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    },
    // Add custom webpack config if needed
    webpack: (config, { isServer }) => {
        // Add any custom webpack configuration here
        return config;
    },
};

module.exports = nextConfig;
