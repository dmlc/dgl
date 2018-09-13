// vector view
#ifndef DGL_VECTOR_VIEW_H_
#define DGL_VECTOR_VIEW_H_

#include <vector>
#include <memory>
#include <stdint.h>

#include <dmlc/logging.h>

namespace dgl {

/*!
 * \brief A vector-like data structure that can be either a vector or a read-only view.
 */
template<typename ValueType>
class vector_view {
 public:
  /*! \brief iterator class */
  class iterator : public std::iterator<std::forward_iterator_tag, ValueType> {
   public:
    /*! \brief iterator constructor */
    iterator(const vector_view<ValueType>* vec, size_t pos): vec_(vec), pos_(pos) {}
    /*! \brief move to next */
    iterator& operator++() {
      ++pos_;
      return *this;
    }
    /*! \brief move to next */
    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }
    /*! \brief equal operator */
    bool operator==(iterator other) const {
      return vec_ == other.vec_ and pos_ == other.pos_;
    }
    /*! \brief not equal operator */
    bool operator!=(iterator other) const {
      return !(*this == other);
    }
    /*! \brief dereference operator */
    const ValueType& operator*() const {
      return (*vec_)[pos_];
    }
   private:
    /*! \brief vector_view pointer */
    const vector_view<ValueType>* vec_;
    /*! \brief current position */
    size_t pos_;
  };

  /*! \brief Default constructor. Create an empty vector. */
  vector_view()
    : data_(std::make_shared<std::vector<ValueType> >()) {}

  /*! \brief Construct a new vector view that shares the data with the given vec. */
  vector_view(const vector_view<ValueType>& vec, const std::vector<uint64_t>& index)
    : data_(vec.data_), index_(index), is_view_(true) {}

  /*! \brief constructor from a vector pointer */
  explicit vector_view(const std::shared_ptr<std::vector<ValueType> >& other)
    : data_(other) {}

  /*! \brief default copy constructor */
  vector_view(const vector_view<ValueType>& other) = default;

#ifndef _MSC_VER
  /*! \brief default move constructor */
  vector_view(vector_view<ValueType>&& other) = default;
#else
  /*! \brief default move constructor */
  vector_view(vector_view<ValueType>&& other) {
    data_ = other.data_;
    index_ = other.index_;
    is_view_ = other.is_view_;
    other.data_ = std::make_shared<std::vector<ValueType> >();
    other.index_ = std::vector<uint64_t>();
    other.is_view_ = false;
  }
#endif  // _MSC_VER

  /*! \brief default destructor */
  ~vector_view() = default;

  /*! \brief default assign constructor */
  vector_view<ValueType>& operator=(const vector_view<ValueType>& other) = default;

  /*! \return the size of the vector */
  size_t size() const {
    if (is_view_) {
      return index_.size();
    } else {
      return data_->size();
    }
  }

  /*!
   * \brief Return the i^th element of the vector.
   * \param i The index
   * \return reference to the i^th element
   */
  ValueType& operator[](size_t i) {
    CHECK(!is_view_);
    return data_[i];
  }

  /*!
   * \brief Return the i^th element of the vector.
   * \param i The index
   * \return reference to the i^th element
   */
  const ValueType& operator[](size_t i) const {
    if (is_view_) {
      return data_[index_[i]];
    } else {
      return data_[i];
    }
  }

  /*! \return an iterator pointing at the first element */
  iterator begin() const {
    return iterator(this, 0);
  }

  /*! \return an iterator pointing at the last element */
  iterator end() const {
    return iterator(this, size());
  }

  // Modifiers
  // NOTE: The modifiers are not allowed for view.
  /*!
   * \brief Add element at the end.
   * \param val The value
   */
  void push_back(const ValueType& val) {
    CHECK(!is_view_);
    data_->push_back(val);
  }

  /*!
   * \brief Clear the vector.
   */
  void clear() {
    CHECK(!is_view_);
    data_ = std::make_shared<std::vector<ValueType> >();
  }

  /*! \brief Resize the vector */
  void resize(size_t size) {
    CHECK(!is_view_);
    data_->resize(size);
  }

  /*! \brief Resize the vector with init value */
  void resize(size_t size, const ValueType& val) {
    CHECK(!is_view_);
    data_->resize(size, val);
  }

 private:
  /*! \brief pointer to the underlying vector data */
  std::shared_ptr<std::vector<ValueType> > data_;
  /*! \brief index used to access the data vector */
  std::vector<uint64_t> index_;
  /*! \brief whether this is a view or a vector */
  bool is_view_{false};
};

}  // namespace dgl

#endif  // DGL_VECTOR_VIEW_H_
