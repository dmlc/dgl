#include <iostream>
#include <vector>
#include <functional>
#include <iterator>
#include <cassert>
using namespace std;

int main( void )
{
    vector<int> a{1,2,3,4,5,6};

    // 引用 a 中的 {3,4,5}
    vector<reference_wrapper<int>> b{ next(begin(a),2), next(begin(a),5) };
    for( const auto& v : b )
        cout << v << '\t';

    // 修改 b[0] 就相当于修改了 a[2]
    b[0].get() = 100; // 就是这里丑，没办法
    for( const auto& v : a )
        cout << v << '\t';

    return 0;
}
