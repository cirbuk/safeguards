import React from 'react';
import {
  Box,
  Grid,
  Heading,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  SimpleGrid,
  useColorModeValue,
} from '@chakra-ui/react';
import { Line } from 'react-chartjs-2';
import Layout from '../components/Layout';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard: React.FC = () => {
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  // Sample data - replace with real API data
  const cpuData = {
    labels: ['1h', '2h', '3h', '4h', '5h', 'Now'],
    datasets: [
      {
        label: 'CPU Usage',
        data: [30, 45, 35, 50, 40, 60],
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  const budgetData = {
    labels: ['1h', '2h', '3h', '4h', '5h', 'Now'],
    datasets: [
      {
        label: 'Budget Usage',
        data: [200, 300, 400, 350, 500, 450],
        borderColor: 'rgb(153, 102, 255)',
        tension: 0.1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <Layout>
      <Box>
        <Heading mb={6}>Dashboard</Heading>

        {/* Stats Overview */}
        <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
          <Stat
            p={4}
            bg={cardBg}
            borderRadius="lg"
            borderWidth="1px"
            borderColor={borderColor}
          >
            <StatLabel>Active Agents</StatLabel>
            <StatNumber>12</StatNumber>
            <StatHelpText>
              <StatArrow type="increase" />
              23.36%
            </StatHelpText>
          </Stat>

          <Stat
            p={4}
            bg={cardBg}
            borderRadius="lg"
            borderWidth="1px"
            borderColor={borderColor}
          >
            <StatLabel>Total Budget Usage</StatLabel>
            <StatNumber>$450.20</StatNumber>
            <StatHelpText>
              <StatArrow type="decrease" />
              9.05%
            </StatHelpText>
          </Stat>

          <Stat
            p={4}
            bg={cardBg}
            borderRadius="lg"
            borderWidth="1px"
            borderColor={borderColor}
          >
            <StatLabel>CPU Usage</StatLabel>
            <StatNumber>45%</StatNumber>
            <StatHelpText>
              <StatArrow type="increase" />
              12.58%
            </StatHelpText>
          </Stat>

          <Stat
            p={4}
            bg={cardBg}
            borderRadius="lg"
            borderWidth="1px"
            borderColor={borderColor}
          >
            <StatLabel>Memory Usage</StatLabel>
            <StatNumber>2.4 GB</StatNumber>
            <StatHelpText>
              <StatArrow type="increase" />
              5.12%
            </StatHelpText>
          </Stat>
        </SimpleGrid>

        {/* Charts */}
        <Grid templateColumns={{ base: '1fr', lg: 'repeat(2, 1fr)' }} gap={6}>
          <Box
            p={6}
            bg={cardBg}
            borderRadius="lg"
            borderWidth="1px"
            borderColor={borderColor}
          >
            <Heading size="md" mb={4}>CPU Usage Trend</Heading>
            <Line options={chartOptions} data={cpuData} />
          </Box>

          <Box
            p={6}
            bg={cardBg}
            borderRadius="lg"
            borderWidth="1px"
            borderColor={borderColor}
          >
            <Heading size="md" mb={4}>Budget Usage Trend</Heading>
            <Line options={chartOptions} data={budgetData} />
          </Box>
        </Grid>
      </Box>
    </Layout>
  );
};

export default Dashboard;
