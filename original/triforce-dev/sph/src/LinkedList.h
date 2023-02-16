#ifndef TRIFORCE_LINKEDLIST_H
#define TRIFORCE_LINKEDLIST_H
/*
 * LinkedList.h
 *
 * Basic linked list class modified from:
 * https://www.tutorialspoint.com/cplusplus-program-to-implement-singly-linked-list
 * https://www.geeksforgeeks.org/implementing-iterator-pattern-of-a-single-linked-list/
 * https://leetcode.com/problems/design-linked-list/ (fastest submission)
 *
 * Created On: 08/04/2022
 *
 * Last Updated:
 *    * AJK - 08/04/2022 - Initially created
 */

#include "Utilities.h"


template<typename T>
class LinkedList {
    public:
        struct node{
            T val;
            node* next;
        };

        LinkedList() : head(nullptr), tail(nullptr), length(0) {};


        /* Get the value of the index-th node in the linked list.*/
        T get(uint32_t index) {
            node* current = head;
            T val = current->val;
            while(index>0 && current){
                current = current->next;
                val = current->val;
                --index;
            }
            return val;
        }

        uint32_t getSize() const {
            return length;
        }

        /* Add a node of value val before the first element of the
         * linked list. After the insertion, the new node will be
         * the first node of the linked list. */
        void addAtHead(T val) {
            node* n = new node{val,head};
            head = n;
            if(length == 0){
                tail = n;
            }
            length++;
        }

        /** Append a node of value val to the last element of the linked list. */
        void addAtTail(T val) {
            node* n = new node{val,nullptr};
            if(tail){
                tail->next = n;
            }

            tail = n;
            if(length==0){
                head = n;
            }
            length++;
        }

        /* Add a node of value val before the index-th node in the linked list.
         * If index equals to the length of linked list, the node will be appended to the
         * end of linked list. If index is greater than the length,
         * the node will not be inserted.*/
        void addAtIndex(uint32_t index, T val) {
            if(index>length){
                return;
            }
            node* previous = nullptr;
            node* current = head;
            while(index > 0 && current){
                previous = current;
                current = current->next;
                --index;
            }
            node* n = new node{val, current};
            if(previous){
                previous->next = n;
            }else{
                head = n;
            }
            if(!current){
                tail = n;
            }
            ++length;
        }

        /** Delete the index-th node in the linked list, if the index is valid. */
        void deleteAtIndex(uint32_t index) {
            if(index >= length){
                return;
            }
            node* previous = nullptr;
            node* current = head;
            while(index > 0 && current){
                previous = current;
                current = current->next;
                --index;
            }
            if(previous){
                previous->next = current->next;
            }else{
                head = current->next;
            }
            if(!current->next){
                tail = previous;
            }
            delete(current);
            --length;
        }

        void deleteAllNodes() {
            if (head == nullptr || length == 0) { // linked list not created yet
                return;
            }
            if (head->next == nullptr) {
                head = nullptr;
                tail = nullptr;
                length--;
                return;
            }
            head = head->next;
            length--;
            deleteAllNodes();
        }

    public:
        node* head;
        node* tail;
        uint32_t length;
};




/*

template<typename T>
class LinkedList {
private: // Member classes
    struct Node;
    using Nodeptr = shared_ptr<Node>;

private: // Private Data
    uint32_t size;
    Nodeptr head;
    Nodeptr tail;

public: // Constructors & Destructor
    LinkedList() : size(0), head(nullptr), tail(nullptr) {};
    explicit LinkedList(T value) : size(1), head(make_shared<Node>(value)), tail(head) {};
    ~LinkedList() = default;

public: // Public functions

    T get(uint32_t idx) {
        if (idx >= size) { return T(std::numeric_limits<T>::lowest()); }
        uint32_t count = 0;
        Nodeptr temp = head;
        while (++count < idx) {
            temp = temp->next;
        }
        return temp->data;
    }

    uint32_t getSize() const {
        return size;
    }

    void addAtHead(T value) {
        Nodeptr new_node = make_shared<Node>(value, head);
        head = new_node;
        if (size == 0) {
            tail = n;
        }
        ++size;
    };

    void addAtTail(T value) {
        Nodeptr new_node = make_shared<Node>(value, nullptr);

        if (tail==nullptr) {
            head = tail = new_node;
        } else {
            tail->next = new_node;
            tail = new_node;
        }
        size++;
    }

    void addAtIndex(uint32_t idx, T value) {
        if (idx >= size) return;
        if (idx == 0){
            addAtHead(value);
            return;
        }

        uint32_t count = 0;
        Nodeptr it = head;
        Nodeptr new_node = make_shared<Node>(value);
        while (++count < idx) {
            it=it->next;
        }
        new_node->next = it->next;
        it->next = new_node;
        size++;
    }

    void deleteAtIndex(uint32_t idx) {
        if (head == nullptr || idx >= size) {
            return;
        } else if (idx == 0) {
            head = head->next;
        } else {
            uint32_t count = 0;
            Nodeptr it = head;
            Nodeptr prev = nullptr;
            while(++count < idx) {
                prev = it;
                it = it->next;
            }
            prev->next = it->next;
            if (it->next == nullptr) {
                tail = prev;
            }
        }
        size--;
    }

    void deleteAllNodes() {
        if (head == nullptr) { // linked list not created yet
            return;
        } else if (head->next == nullptr) {
            head = nullptr;
            tail = nullptr;
            size--;
            assert(size == 0);
            return;
        }
        head = head->next;
        size--;
        deleteAllNodes();
    }

    T pop() {
        T tmp = get(0);
        deleteAtIndex(0);
        return tmp;
    }

private: // Helper class definitions
    struct Node {
        T data;
        Nodeptr next;

        Node() : data(T()), next(nullptr) {};
        Node(T data, Nodeptr next) : data(data), next(next) {};
        explicit Node(T data) : data(data), next(nullptr) {};
    };

};
*/

using LinkedListFP = LinkedList<fptype>;
using LinkedListUINT = LinkedList<uint32_t>;
using LinkedListINT = LinkedList<int32_t>;
#endif //TRIFORCE_LINKEDLIST_H
