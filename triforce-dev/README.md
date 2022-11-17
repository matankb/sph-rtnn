# TriForce - A framework for multi-physics simulations

## Developers: 
  - Ayden Kish (akis@lle.rochester.edu)
  - Michael Lavell ()
  - Robert Masti ()
  - Andrew Sexton (asexton2@ur.rochester.edu)

*Principal Investigator*: Adam Sefkow (adam.sefkow@rochester.edu)

### Style Guide Preferences
Andrew's Preferences
- Indent four spaces (or one tab)
- \* and & are right associative so they stick to the name, not the type
  - eg. ```float *a, *b, *c;```
- Use spaces between binary ops (and lots of parens)
  - ```C++
    for (int i = 0; i < N; i++) {
        // Good
    }
    
    for(int i=0;i<N;i++){
        // Bad
    }
    
    // Good
    a[idx] = ((b + c) / d) * (e + h);
    
    // Bad
    a[idx]=b+c/d*e+h;
    ```
- Use brackets for `if`/`for` statements, even single lines
  - ```C++
    // Good
    if (thing) { do stuff; }
    
    // Bad
    if (thing) do stuff;    
    ```
- Function arg order 
  - Non-const
  - Const
  - Scalars (since all default arguments must come last)
  - eg. 
    ```C++ 
    foo(float *a, const float *b, const int c, const int d = 0, float e = 1.0)
    ```
- Line breaks and brackets in function calls/definitions
  - ```C++
    // Short parameter lists, bracket is in line with top
    void bar(int a) {
        // stuff
    }
    
    // Long parameter lists, bracket on new line
    void foo(float *a,
             const float *b,
             const int c,
             const int d = 0,
             float e = 1.0)
    {
        // stuff
    }
    ```
  - ```C++
    // Break up args in whatever logical order
    baz(output,
        input1, input2,
        input3, input4,
        extras);
    bop(output,
        input1, input2, input3, input4,
        extras);
    ```
- Avoid C style casts
  - Bad: ```(float) a```
  - Better: ```float(a)```
  - Best: ```static_cast<float>(a)```
  - Exception: Pointer casts ```(float *) &thing[offset]```, but can also use ```reinterpret_cast<float *>(&thing[offset])```