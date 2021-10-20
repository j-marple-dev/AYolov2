/// @file
/// @author Jongkuk Lim <limjk@jmarple.ai>
/// @copyright 2021 J.Marple
/// @brief Unit testing for second_module

#include <gtest/gtest.h>

#include <string>

#include "trt_loader/trt_loader.hpp"

TEST(SecondModuleTest, GenerateWorldTest) {
  std::string test_result1 = second_module::demo_generate_world(1);
  std::string test_result2 = second_module::demo_generate_world(2);
  std::string test_result3 = second_module::demo_generate_world(3);

  EXPECT_EQ(test_result1, "world");
  EXPECT_EQ(test_result2, "worldworld");
  EXPECT_EQ(test_result3, "worldworldworld");
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  auto test_results = RUN_ALL_TESTS();

  return test_results;
}
