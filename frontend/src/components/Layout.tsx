import React from 'react';
import {
  Box,
  Flex,
  VStack,
  IconButton,
  useColorMode,
  useColorModeValue,
} from '@chakra-ui/react';
import { FiMenu, FiSun, FiMoon } from 'react-icons/fi';
import Sidebar from './Sidebar';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [isSidebarOpen, setSidebarOpen] = React.useState(true);
  const { colorMode, toggleColorMode } = useColorMode();
  const bgColor = useColorModeValue('gray.50', 'gray.900');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Flex h="100vh">
      {/* Sidebar */}
      <Box
        w={isSidebarOpen ? '250px' : '0'}
        bg={useColorModeValue('white', 'gray.800')}
        borderRight="1px"
        borderColor={borderColor}
        transition="width 0.2s"
        overflow="hidden"
      >
        <Sidebar />
      </Box>

      {/* Main Content */}
      <VStack flex="1" spacing={0}>
        {/* Header */}
        <Flex
          w="full"
          h="60px"
          px={4}
          bg={useColorModeValue('white', 'gray.800')}
          borderBottom="1px"
          borderColor={borderColor}
          align="center"
          justify="space-between"
        >
          <IconButton
            aria-label="Toggle Sidebar"
            icon={<FiMenu />}
            onClick={() => setSidebarOpen(!isSidebarOpen)}
            variant="ghost"
          />
          <IconButton
            aria-label="Toggle Color Mode"
            icon={colorMode === 'light' ? <FiMoon /> : <FiSun />}
            onClick={toggleColorMode}
            variant="ghost"
          />
        </Flex>

        {/* Content */}
        <Box
          flex="1"
          w="full"
          p={6}
          bg={bgColor}
          overflowY="auto"
        >
          {children}
        </Box>
      </VStack>
    </Flex>
  );
};

export default Layout;
