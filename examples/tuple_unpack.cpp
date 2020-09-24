#include <iostream>
#include <tuple>

using namespace std;

tuple<int, int, int, int> ok(int a, int b) { return {a/2, b/2, a, b}; }

int main() {
    auto [value1, value2, value3, value4] = ok(1, 2);
    cout << value1 << ", " << value2 << ", " << value3 << ", " << value4 << endl;
}