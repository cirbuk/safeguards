import React from 'react';
import {
  VStack,
  Link,
  Text,
  Icon,
  Flex,
  useColorModeValue,
} from '@chakra-ui/react';
import {
  FiHome,
  FiCpu,
  FiDollarSign,
  FiUsers,
  FiSettings,
  FiAlertCircle,
} from 'react-icons/fi';
import NextLink from 'next/link';
import { useRouter } from 'next/router';

interface NavItemProps {
  icon: any;
  children: React.ReactNode;
  href: string;
}

const NavItem: React.FC<NavItemProps> = ({ icon, children, href }) => {
  const router = useRouter();
  const isActive = router.pathname === href;
  const activeBg = useColorModeValue('gray.100', 'gray.700');
  const hoverBg = useColorModeValue('gray.100', 'gray.700');

  return (
    <Link
      as={NextLink}
      href={href}
      style={{ textDecoration: 'none' }}
      _focus={{ boxShadow: 'none' }}
    >
      <Flex
        align="center"
        p="4"
        mx="4"
        borderRadius="lg"
        role="group"
        cursor="pointer"
        bg={isActive ? activeBg : 'transparent'}
        _hover={{
          bg: hoverBg,
        }}
      >
        <Icon
          mr="4"
          fontSize="16"
          as={icon}
        />
        <Text>{children}</Text>
      </Flex>
    </Link>
  );
};

const Sidebar: React.FC = () => {
  return (
    <VStack align="stretch" py="5" h="full">
      <Text
        px="8"
        mb="4"
        fontSize="lg"
        fontWeight="bold"
        color={useColorModeValue('gray.700', 'gray.200')}
      >
        Agent Safety
      </Text>

      <NavItem icon={FiHome} href="/">
        Dashboard
      </NavItem>

      <NavItem icon={FiCpu} href="/metrics">
        Metrics
      </NavItem>

      <NavItem icon={FiDollarSign} href="/budget">
        Budget Control
      </NavItem>

      <NavItem icon={FiUsers} href="/agents">
        Agents
      </NavItem>

      <NavItem icon={FiAlertCircle} href="/alerts">
        Alerts
      </NavItem>

      <NavItem icon={FiSettings} href="/settings">
        Settings
      </NavItem>
    </VStack>
  );
};

export default Sidebar;
