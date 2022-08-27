#!/usr/bin/env perl
use strict;
use warnings;

foreach my $lang ('english', 'dutch') {
    my $raw_path = "./$lang/raw.txt";
    my $cooked_path = "./$lang/all.data";
    open RAW, "< $raw_path" or die "bad open of $raw_path";
    open COOKED, "> $cooked_path" or die "bad open of $cooked_path";
    foreach my $hyphenated_word (<RAW>) {
        chomp $hyphenated_word;
        my %hyphen_index = ();
        my @letter = ();
        my @char = split(//, $hyphenated_word);
        my $hyphen_count = 0;
        for (my $i = 0; $i < scalar(@char); $i++) {
            if ($char[$i] eq '-') {
                $hyphen_count++;
                $hyphen_index{$i - $hyphen_count} = 1;
            } else {
                push(@letter, $char[$i])
            }
        }
        for (my $i = 0; $i < scalar(@letter); $i++) {
            my $hyphen_tag = '*';
            if ($hyphen_index{$i}) {
                $hyphen_tag = '-'
            }
            my $output_row = "$letter[$i] $hyphen_tag\n";
            print COOKED $output_row;
        }
        print COOKED "\n";
    }
    close RAW or die "bad close";
    close COOKED or die "bad close";
}