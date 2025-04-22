import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Button,
  VStack,
  HStack,
  Progress,
  Badge,
  useColorModeValue,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  FormControl,
  FormLabel,
  Input,
  NumberInput,
  NumberInputField,
  useDisclosure,
  useToast,
} from '@chakra-ui/react';
import { FiPlus, FiEdit2, FiTrash2 } from 'react-icons/fi';
import Layout from '../components/Layout';
import { metricsApi } from '../api/metrics';

interface BudgetPool {
  id: string;
  name: string;
  size: number;
  used: number;
  agents: string[];
  priority: number;
}

const BudgetControl: React.FC = () => {
  const [pools, setPools] = useState<BudgetPool[]>([]);
  const [selectedPool, setSelectedPool] = useState<BudgetPool | null>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  useEffect(() => {
    // Load budget pools
    const loadPools = async () => {
      try {
        const response = await fetch('/api/budget/pools');
        const data = await response.json();
        setPools(data);
      } catch (error) {
        console.error('Failed to load budget pools:', error);
        toast({
          title: 'Error loading budget pools',
          status: 'error',
          duration: 5000,
        });
      }
    };

    loadPools();
  }, []);

  const handleCreatePool = async (formData: any) => {
    try {
      const response = await fetch('/api/budget/pools', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        toast({
          title: 'Budget pool created',
          status: 'success',
          duration: 3000,
        });
        // Refresh pools
        const data = await response.json();
        setPools([...pools, data]);
        onClose();
      }
    } catch (error) {
      console.error('Failed to create budget pool:', error);
      toast({
        title: 'Error creating budget pool',
        status: 'error',
        duration: 5000,
      });
    }
  };

  return (
    <Layout>
      <Box>
        <HStack justify="space-between" mb={6}>
          <Heading>Budget Control</Heading>
          <Button
            leftIcon={<FiPlus />}
            colorScheme="blue"
            onClick={onOpen}
          >
            Create Pool
          </Button>
        </HStack>

        {/* Budget Pools Table */}
        <Box
          borderWidth="1px"
          borderRadius="lg"
          borderColor={borderColor}
          overflow="hidden"
        >
          <Table variant="simple">
            <Thead>
              <Tr>
                <Th>Pool Name</Th>
                <Th>Size</Th>
                <Th>Usage</Th>
                <Th>Priority</Th>
                <Th>Agents</Th>
                <Th>Actions</Th>
              </Tr>
            </Thead>
            <Tbody>
              {pools.map((pool) => (
                <Tr key={pool.id}>
                  <Td>{pool.name}</Td>
                  <Td>${pool.size.toFixed(2)}</Td>
                  <Td>
                    <VStack align="start" spacing={1}>
                      <Progress
                        value={(pool.used / pool.size) * 100}
                        size="sm"
                        width="100%"
                        colorScheme={
                          (pool.used / pool.size) * 100 > 80
                            ? 'red'
                            : (pool.used / pool.size) * 100 > 60
                            ? 'yellow'
                            : 'green'
                        }
                      />
                      <Box fontSize="sm">
                        ${pool.used.toFixed(2)} / ${pool.size.toFixed(2)}
                      </Box>
                    </VStack>
                  </Td>
                  <Td>
                    <Badge
                      colorScheme={
                        pool.priority === 1
                          ? 'red'
                          : pool.priority === 2
                          ? 'orange'
                          : 'green'
                      }
                    >
                      P{pool.priority}
                    </Badge>
                  </Td>
                  <Td>{pool.agents.length}</Td>
                  <Td>
                    <HStack spacing={2}>
                      <Button
                        size="sm"
                        leftIcon={<FiEdit2 />}
                        onClick={() => {
                          setSelectedPool(pool);
                          onOpen();
                        }}
                      >
                        Edit
                      </Button>
                      <Button
                        size="sm"
                        leftIcon={<FiTrash2 />}
                        colorScheme="red"
                        variant="ghost"
                      >
                        Delete
                      </Button>
                    </HStack>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Box>

        {/* Create/Edit Pool Modal */}
        <Modal isOpen={isOpen} onClose={onClose}>
          <ModalOverlay />
          <ModalContent>
            <ModalHeader>
              {selectedPool ? 'Edit Budget Pool' : 'Create Budget Pool'}
            </ModalHeader>
            <ModalBody>
              <VStack spacing={4}>
                <FormControl>
                  <FormLabel>Pool Name</FormLabel>
                  <Input
                    placeholder="Enter pool name"
                    defaultValue={selectedPool?.name}
                  />
                </FormControl>
                <FormControl>
                  <FormLabel>Pool Size</FormLabel>
                  <NumberInput
                    defaultValue={selectedPool?.size || 1000}
                    min={100}
                  >
                    <NumberInputField />
                  </NumberInput>
                </FormControl>
                <FormControl>
                  <FormLabel>Priority</FormLabel>
                  <NumberInput
                    defaultValue={selectedPool?.priority || 3}
                    min={1}
                    max={3}
                  >
                    <NumberInputField />
                  </NumberInput>
                </FormControl>
              </VStack>
            </ModalBody>
            <ModalFooter>
              <Button variant="ghost" mr={3} onClick={onClose}>
                Cancel
              </Button>
              <Button
                colorScheme="blue"
                onClick={() => {
                  // Handle form submission
                  handleCreatePool({
                    name: 'New Pool',
                    size: 1000,
                    priority: 3,
                  });
                }}
              >
                {selectedPool ? 'Save Changes' : 'Create Pool'}
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </Box>
    </Layout>
  );
};

export default BudgetControl;
