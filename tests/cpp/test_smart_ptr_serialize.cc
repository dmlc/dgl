#include <dgl/runtime/serializer.h>
#include <dgl/runtime/smart_ptr_serializer.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <dmlc/parameter.h>
#include <gtest/gtest.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <unordered_map>

using namespace std;

class MyClass {
 public:
  MyClass() {}
  MyClass(std::string data) : data_(data) {}
  inline void Save(dmlc::Stream *strm) const { strm->Write(this->data_); }
  inline bool Load(dmlc::Stream *strm) { return strm->Read(&data_); }
  inline bool operator==(const MyClass &other) const {
    return data_ == other.data_;
  }

 public:
  std::string data_;
};
// need to declare the traits property of my class to dmlc
namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, MyClass, true);
}

template <typename T>
class SmartPtrTest : public ::testing::Test {
 public:
  typedef T SmartPtr;
};

using SmartPtrTypes =
    ::testing::Types<std::shared_ptr<MyClass>, std::unique_ptr<MyClass>>;
TYPED_TEST_SUITE(SmartPtrTest, SmartPtrTypes);

TYPED_TEST(SmartPtrTest, Obj_Test) {
  std::string blob;
  dmlc::MemoryStringStream fs(&blob);
  using SmartPtr = typename TestFixture::SmartPtr;
  auto myc = SmartPtr(new MyClass("1111"));
  { static_cast<dmlc::Stream *>(&fs)->Write(myc); }
  fs.Seek(0);
  auto copy_data = SmartPtr(new MyClass());
  CHECK(static_cast<dmlc::Stream *>(&fs)->Read(&copy_data));

  EXPECT_EQ(myc->data_, copy_data->data_);
}

TYPED_TEST(SmartPtrTest, Vector_Test1) {
  std::string blob;
  dmlc::MemoryStringStream fs(&blob);
  using SmartPtr = typename TestFixture::SmartPtr;
  typedef std::pair<std::string, SmartPtr> Pair;

  std::vector<Pair> myclasses;
  myclasses.emplace_back("a", SmartPtr(new MyClass("@A@B")));
  myclasses.emplace_back("b", SmartPtr(new MyClass("2222")));
  static_cast<dmlc::Stream *>(&fs)->Write<std::vector<Pair>>(myclasses);

  dmlc::MemoryStringStream ofs(&blob);
  std::vector<Pair> copy_myclasses;
  static_cast<dmlc::Stream *>(&ofs)->Read<std::vector<Pair>>(&copy_myclasses);

  EXPECT_TRUE(std::equal(
      myclasses.begin(), myclasses.end(), copy_myclasses.begin(),
      [](const Pair &left, const Pair &right) {
        return (left.second->data_ == right.second->data_) &&
               (left.first == right.first);
      }));
}

TYPED_TEST(SmartPtrTest, Vector_Test2) {
  std::string blob;
  dmlc::MemoryStringStream fs(&blob);
  using SmartPtr = typename TestFixture::SmartPtr;

  std::vector<SmartPtr> myclasses;
  myclasses.emplace_back(new MyClass("@A@"));
  myclasses.emplace_back(new MyClass("2222"));
  static_cast<dmlc::Stream *>(&fs)->Write<std::vector<SmartPtr>>(myclasses);

  dmlc::MemoryStringStream ofs(&blob);
  std::vector<SmartPtr> copy_myclasses;
  static_cast<dmlc::Stream *>(&ofs)->Read<std::vector<SmartPtr>>(
      &copy_myclasses);

  EXPECT_TRUE(std::equal(
      myclasses.begin(), myclasses.end(), copy_myclasses.begin(),
      [](const SmartPtr &left, const SmartPtr &right) {
        return left->data_ == right->data_;
      }));
}
