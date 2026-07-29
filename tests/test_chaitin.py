"""
Unit Tests for Chaitin-Briggs Allocator Baseline.
"""

import unittest
from compiler.ir import Program, Instruction, OpCode
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph
from compiler.chaitin_briggs import ChaitinBriggsAllocator


class TestChaitinBriggsAllocator(unittest.TestCase):
    def setUp(self):
        self.prog = Program(name="coalesce_prog")
        v0 = self.prog.get_or_create_var("v0")
        v1 = self.prog.get_or_create_var("v1")
        v2 = self.prog.get_or_create_var("v2")
        c = self.prog.get_or_create_var("const_10", is_const=True, const_val=10)

        self.prog.add_instruction(Instruction(OpCode.ASSIGN, target=v0, arg1=c))
        self.prog.add_instruction(Instruction(OpCode.MOVE, target=v1, arg1=v0))
        self.prog.add_instruction(Instruction(OpCode.ADD, target=v2, arg1=v1, arg2=c))
        self.prog.add_instruction(Instruction(OpCode.RETURN, arg1=v2))

    def test_chaitin_allocation_and_coalescing(self):
        cfg = ControlFlowGraph(self.prog)
        liveness = LivenessAnalyzer(cfg)
        ig = InterferenceGraph(self.prog, cfg, liveness)

        allocator = ChaitinBriggsAllocator(num_registers=4)
        result = allocator.allocate(ig)

        self.assertTrue(len(result.register_assignment) > 0)
        self.assertEqual(len(result.spilled_vars), 0)


if __name__ == "__main__":
    unittest.main()
