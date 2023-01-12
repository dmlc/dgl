/**
 *  Copyright (c) 2019 by Contributors
 * @file string_test.cc
 * @brief Test String Common
 */
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../src/rpc/network/common.h"

using dgl::network::SplitStringUsing;
using dgl::network::SStringPrintf;
using dgl::network::StringAppendF;
using dgl::network::StringPrintf;

TEST(SplitStringTest, SplitStringUsingCompoundDelim) {
  std::string full(" apple \torange ");
  std::vector<std::string> subs;
  SplitStringUsing(full, " \t", &subs);
  EXPECT_EQ(subs.size(), 2);
  EXPECT_EQ(subs[0], std::string("apple"));
  EXPECT_EQ(subs[1], std::string("orange"));
}

TEST(SplitStringTest, testSplitStringUsingSingleDelim) {
  std::string full(" apple orange ");
  std::vector<std::string> subs;
  SplitStringUsing(full, " ", &subs);
  EXPECT_EQ(subs.size(), 2);
  EXPECT_EQ(subs[0], std::string("apple"));
  EXPECT_EQ(subs[1], std::string("orange"));
}

TEST(SplitStringTest, testSplitingNoDelimString) {
  std::string full("apple");
  std::vector<std::string> subs;
  SplitStringUsing(full, " ", &subs);
  EXPECT_EQ(subs.size(), 1);
  EXPECT_EQ(subs[0], std::string("apple"));
}

TEST(StringPrintf, normal) {
  using std::string;
  EXPECT_EQ(StringPrintf("%d", 1), string("1"));
  string target;
  SStringPrintf(&target, "%d", 1);
  EXPECT_EQ(target, string("1"));
  StringAppendF(&target, "%d", 2);
  EXPECT_EQ(target, string("12"));
}
