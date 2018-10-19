#!/usr/bin/env groovy

def setup() {
    sh 'easy_install nose'
    sh 'git submodule init'
    sh 'git submodule update'
}

def build_dgl() {
    sh 'if [ -d build ]; then rm -rf build; fi; mkdir build'
    dir('python') {
        sh 'python3 setup.py install'
    }
    dir ('build') {
        sh 'cmake ..'
        sh 'make -j$(nproc)'
    }
}

def unit_test() {
    withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build"]) {
        sh 'nosetests tests -v --with-xunit'
        sh 'nosetests tests/pytorch -v --with-xunit'
        sh 'nosetests tests/graph_index -v --with-xunit'
    }
}

def example_test(dev) {
    dir ('tests/scripts') {
        withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build"]) {
            sh "./test_examples_${dev}.sh"
        }
    }
}

pipeline {
    agent none
    stages {
        stage('Build and Test') {
            parallel {
                stage('CPU') {
                    agent {
                        docker {
                            image 'lingfanyu/dgl-cpu'
                            args '-u root'
                        }
                    }
                    stages {
                        stage('SETUP') {
                            steps {
                                setup()
                            }
                        }
                        stage('BUILD') {
                            steps {
                                build_dgl()
                            }
                        }
                        stage('UNIT TEST') {
                            steps {
                                unit_test()
                            }
                        }
                        stage('EXAMPLE TEST') {
                            steps {
                                example_test('cpu')
                            }
                        }
                    }
                    post {
                        always {
                            junit '*.xml'
                        }
                    }
                }
                stage('GPU') {
                    agent {
                        docker {
                            image 'lingfanyu/dgl-gpu'
                            args '--runtime nvidia -u root'
                        }
                    }
                    stages {
                        stage('SETUP') {
                            steps {
                                setup()
                            }
                        }
                        stage('BUILD') {
                            steps {
                                build_dgl()
                            }
                        }
                        stage('UNIT TEST') {
                            steps {
                                unit_test()
                            }
                        }
                        stage('EXAMPLE TEST') {
                            steps {
                                example_test('gpu')
                            }
                        }
                    }
                    post {
                        always {
                            junit '*.xml'
                        }
                    }
                }
            }
        }
    }
}
