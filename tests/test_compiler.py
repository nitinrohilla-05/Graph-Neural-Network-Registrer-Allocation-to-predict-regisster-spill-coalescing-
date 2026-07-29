"""
Unit Tests for Compiler Subsystem (IR, CFG, Liveness, Interference Graph).
"""

import unittest
from compiler.ir import Program, Instruction, OpCode
from compiler.cfg import ControlFlowGraph
from compiler.liveness import LivenessAnalyzer
from compiler.interference_graph import InterferenceGraph


class TestCompilerSubsystem(unittest.TestCase):
    def setUp(self):
        self.prog = Program(name="test_prog")
        v0 = self.prog.get_or_create_var("v0")
        v1 = self.prog.get_or_create_var("v1")
        v2 = self.prog.get_or_create_var("v2")
        c1 = self.prog.get_or_create_var("const_1", is_const=True, const_val=1)
        c5 = self.prog.get_or_create_var("const_5", is_const=True, const_val=5)

        self.prog.add_instruction(Instruction(OpCode.ASSIGN, target=v0, arg1=c1))
        self.prog.add_instruction(Instruction(OpCode.ASSIGN, target=v1, arg1=c5))
        self.prog.add_instruction(Instruction(OpCode.ADD, target=v2, arg1=v0, arg2=v1))
        self.prog.add_instruction(Instruction(OpCode.RETURN, arg1=v2))

    def test_cfg_construction(self):
        cfg = ControlFlowGraph(self.prog)
        self.assertTrue(len(cfg.blocks) > 0)
        self.assertEqual(cfg.entry_block.block_id, 0)

    def test_liveness_analysis(self):
        cfg = ControlFlowGraph(self.prog)
        liveness = LivenessAnalyzer(cfg)
        self.assertTrue(len(liveness.inst_live_out) > 0)

    def test_interference_graph_building(self):
        cfg = ControlFlowGraph(self.prog)
        liveness = LivenessAnalyzer(cfg)
        ig = InterferenceGraph(self.prog, cfg, liveness)

        self.assertEqual(len(ig.variables), 3)
        features = ig.get_node_features()
        self.assertIn("v0", features)
        self.assertIn("v1", features)
        self.assertIn("v2", features)

        feat_matrix = ig.get_feature_matrix()
        self.assertEqual(feat_matrix.shape, (3, 6))


if __name__ == "__main__":
    unittest.main()
